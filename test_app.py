import asyncio
import json
import sqlite3
import struct
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
import app as module


def ai_result(text):
    return module.raw.types.messages.ComposedRichMessageWithAI(result=module.raw.types.RichMessage(
        blocks=[module.raw.types.PageBlockPreformatted(text=module.raw.types.TextPlain(text=text), language='json')],
        photos=[], documents=[]))


def engine(tmp_path, fail=False):
    invoke = AsyncMock(side_effect=RuntimeError('failure')) if fail else AsyncMock(return_value=ai_result('{"text":"answer","memory":null}'))
    client = SimpleNamespace(invoke=invoke, is_connected=True)
    return module.Engine(client, module.Store(str(tmp_path / 'db.sqlite')), prompt_limit=12000, use_context=True)


def test_context_persistence_and_isolation(tmp_path):
    e = engine(tmp_path)
    asyncio.run(e.generate('a', 'hello'))
    asyncio.run(e.generate('a', 'next'))
    assert len(e.store.get('a')) == 4
    assert e.store.get('b') == []
    sent = e.client.invoke.call_args.args[0]
    assert 'hello' in sent.tone.custom_prompt and 'next' in sent.tone.custom_prompt
    assert sent.tone.custom_prompt.startswith(module.INSTRUCTION)
    assert isinstance(sent, module.raw.functions.messages.ComposeRichMessageWithAI)
    assert sent.text is None and not sent.proofread
    e.store.delete('a')
    assert e.store.get('a') == []


def test_failure_does_not_commit(tmp_path):
    e = engine(tmp_path, fail=True)
    e.store.put('a', [], 'existing memory')
    with pytest.raises(RuntimeError):
        asyncio.run(e.generate('a', 'hello'))
    assert e.store.get('a') == []
    assert e.store.memory('a') == 'existing memory'


def test_context_bound():
    turns = [{'role': 'user', 'content': 'x' * 6000}, {'role': 'assistant', 'content': 'x' * 6000}, {'role': 'user', 'content': 'new'}]
    result = module.payload('system', turns)
    assert 'new' in result and len(result) <= module.MAX_CONTEXT
    with pytest.raises(ValueError):
        module.payload('s', [{'role': 'user', 'content': 'x' * 20000}])


def test_guest_answer_and_allowlist(tmp_path):
    async def run():
        e = engine(tmp_path)
        bot = module.Bot('fake', e, {42})
        bot.me = {'id': 7, 'username': 'testbot'}
        bot.call = AsyncMock()
        message = {'guest_query_id': 'g1', 'message_id': 3, 'chat': {'id': 8, 'type': 'group'}, 'from': {'id': 42}, 'text': '@testbot hi'}
        await bot.handle({'guest_message': message})
        assert bot.call.call_args.args[0] == 'answerGuestQuery'
        assert bot.call.call_args.kwargs['guest_query_id'] == 'g1'
        assert bot.call.call_args.kwargs['result']['input_message_content'] == {'rich_message': {'markdown': 'answer'}}
        bot.call.reset_mock()
        message['from']['id'] = 99
        await bot.handle({'guest_message': message})
        bot.call.assert_not_called()
        await bot.http.aclose()
    asyncio.run(run())


def test_http_auth_context_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv('API_KEY', 'a' * 32)
    module.app.state.engine = engine(tmp_path)
    module.app.state.bot_task = None
    async def run():
        async with module.httpx.AsyncClient(transport=module.httpx.ASGITransport(app=module.app), base_url='http://test', trust_env=False) as client:
            assert (await client.get('/health')).status_code == 401
            headers = {'Authorization': 'Bearer ' + 'a' * 32}
            response = await client.post('/generate', headers=headers, json={'prompt': 'hi', 'conversation_id': 'test'})
            assert response.status_code == 200 and response.json()['text'] == 'answer'
            assert response.json()['format'] == 'rich_markdown'
            assert len((await client.get('/context/test', headers=headers)).json()['messages']) == 2
            assert (await client.delete('/context/test', headers=headers)).status_code == 200
            assert (await client.get('/context/test', headers=headers)).json()['messages'] == []
    asyncio.run(run())


def test_memory_survives_restart_reset_and_history_trimming(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'MAX_TURNS', 1)
    e = engine(tmp_path)
    e.client.invoke.return_value.result.blocks[0].text.text = json.dumps({'text': 'Запомнил.', 'memory': 'Любит короткие ответы.'})
    assert asyncio.run(e.generate('a', 'Запомни: люблю короткие ответы')) == 'Запомнил.'
    assert e.store.memory('a') == 'Любит короткие ответы.'
    assert e.store.get('a')[-1]['content'] == 'Запомнил.'
    e.store.db.close()
    e = engine(tmp_path)
    asyncio.run(e.generate('a', 'next'))
    assert len(e.store.get('a')) == 2
    assert e.store.memory('a') == 'Любит короткие ответы.'
    e.store.delete('a')
    asyncio.run(e.generate('a', 'what do you remember?'))
    sent = json.loads(e.client.invoke.call_args.args[0].tone.custom_prompt.removeprefix(module.INSTRUCTION))
    assert sent['memory'] == 'Любит короткие ответы.'
    assert len(sent['messages']) == 1
    asyncio.run(e.generate('b', 'hello'))
    assert json.loads(e.client.invoke.call_args.args[0].tone.custom_prompt.removeprefix(module.INSTRUCTION))['memory'] == ''
    assert e.store.memory('b') == ''


def test_existing_database_upgrade_preserves_history_and_offset(tmp_path):
    path = tmp_path / 'old.sqlite'
    turns = [{'role': 'user', 'content': 'existing conversation'}]
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE conversations (id TEXT PRIMARY KEY, turns TEXT NOT NULL)')
        db.execute('CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('INSERT INTO conversations VALUES (?,?)', ('a', json.dumps(turns)))
        db.execute("INSERT INTO state VALUES ('offset','123')")
    db.close()
    store = module.Store(path)
    assert store.get('a') == turns and store.offset() == 123
    assert store.memory('a') == ''
    store.put('a', turns, 'new memory')
    assert store.memory('a') == 'new memory'
    store.db.close()


@pytest.mark.parametrize('response', [
    'plain text', '{"text":', '{"text":"ok","memory":42}',
    '{"text":"ok","memory":"x","conversation_id":"other"}',
    '{"text":"   ","memory":"x"}',
    json.dumps({'text': 'ok', 'memory': 'x' * 3001}),
])
def test_invalid_ai_reply_preserves_memory_and_history(tmp_path, response):
    e = engine(tmp_path)
    turns = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}]
    e.store.put('a', turns, 'old memory')
    e.client.invoke.return_value.result.blocks[0].text.text = response
    with pytest.raises(module.InvalidReply):
        asyncio.run(e.generate('a', 'new'))
    assert e.store.memory('a') == 'old memory'
    assert e.store.get('a') == turns


def test_model_can_replace_and_clear_memory(tmp_path):
    e = engine(tmp_path)
    e.store.put('a', [], 'old')
    for memory in ('new', ''):
        e.client.invoke.return_value.result.blocks[0].text.text = '```json\n' + json.dumps({'text': 'ok', 'memory': memory}) + '\n```'
        asyncio.run(e.generate('a', 'update'))
        assert e.store.memory('a') == memory


def test_memory_and_history_commit_atomically(tmp_path):
    e = engine(tmp_path)
    e.store.put('a', [], 'old')
    e.store.db.execute("CREATE TRIGGER fail_memory BEFORE INSERT ON memories BEGIN SELECT RAISE(ABORT, 'test'); END")
    e.client.invoke.return_value.result.blocks[0].text.text = '{"text":"ok","memory":"new"}'
    with pytest.raises(sqlite3.IntegrityError):
        asyncio.run(e.generate('a', 'new'))
    assert e.store.get('a') == []
    assert e.store.memory('a') == 'old'


def test_memory_counts_toward_context_limit(monkeypatch):
    monkeypatch.setattr(module, 'MAX_CONTEXT', 1000)
    turns = [{'role': 'user', 'content': 'x' * 400}, {'role': 'assistant', 'content': 'y' * 400}, {'role': 'user', 'content': 'new'}]
    result = module.payload('s', turns, 'facts' * 30)
    assert len(result) <= 1000
    assert json.loads(result.removeprefix(module.INSTRUCTION))['messages'] == turns[-1:]
    assert json.loads(result.removeprefix(module.INSTRUCTION))['memory'] == 'facts' * 30
    with pytest.raises(ValueError):
        module.payload('s', turns[-1:], 'facts' * 300)


def test_guest_memory_commands_are_scoped_and_protected(tmp_path):
    async def run():
        e = engine(tmp_path)
        bot = module.Bot('fake', e, {42, 43})
        bot.me = {'id': 7, 'username': 'testbot'}
        bot.call = AsyncMock()
        key = 'tg:guest:8:0:42'
        e.store.put(key, [{'role': 'user', 'content': 'old'}], 'private fact')
        message = {'guest_query_id': 'g1', 'message_id': 3, 'chat': {'id': 8, 'type': 'group'}, 'from': {'id': 42}, 'text': '@testbot /memory'}
        await bot.handle({'guest_message': message})
        assert bot.call.call_args.kwargs['result']['input_message_content']['message_text'] == 'private fact'
        for change in ({'from': {'id': 43}}, {'chat': {'id': 9, 'type': 'group'}}, {'message_thread_id': 5}, {'guest_query_id': None}):
            other = message | change
            await bot.handle({'guest_message': other})
            data = bot.call.call_args.kwargs
            assert (data['result']['input_message_content']['message_text'] if other.get('guest_query_id') else data['text']) == 'Память пока пуста.'
        bot.call.reset_mock()
        await bot.handle({'guest_message': message | {'from': {'id': 99}, 'text': '@testbot /forget'}})
        bot.call.assert_not_called()
        assert e.store.memory(key) == 'private fact'
        message['text'] = '@testbot /reset'
        await bot.handle({'guest_message': message})
        assert e.store.get(key) == [] and e.store.memory(key) == 'private fact'
        message['text'] = '/forget@testbot'
        async with e.lock:
            await bot.handle({'guest_message': message})
        assert e.store.memory(key) == 'private fact'
        await bot.handle({'guest_message': message})
        assert e.store.get(key) == [] and e.store.memory(key) == ''
        e.client.invoke.assert_not_called()
        await bot.http.aclose()
    asyncio.run(run())


def test_memory_http_auth_forget_and_invalid_reply(tmp_path, monkeypatch):
    monkeypatch.setenv('API_KEY', 'a' * 32)
    e = engine(tmp_path)
    monkeypatch.setattr(module.app.state, 'engine', e, raising=False)
    e.store.put('api:test', [{'role': 'user', 'content': 'old'}], 'fact')
    e.store.put('api:other', [], 'other fact')
    async def run():
        async with module.httpx.AsyncClient(transport=module.httpx.ASGITransport(app=module.app), base_url='http://test', trust_env=False) as client:
            assert (await client.get('/memory/test')).status_code == 401
            assert (await client.delete('/memory/test')).status_code == 401
            headers = {'Authorization': 'Bearer ' + 'a' * 32}
            assert (await client.get('/memory/test', headers=headers)).json()['memory'] == 'fact'
            async with e.lock:
                assert (await client.delete('/memory/test', headers=headers)).status_code == 409
            e.client.invoke.return_value.result.blocks[0].text.text = 'bad reply'
            response = await client.post('/generate', headers=headers, json={'prompt': 'hi', 'conversation_id': 'test'})
            assert response.status_code == 502
            assert e.store.memory('api:test') == 'fact'
            assert (await client.delete('/memory/test', headers=headers)).status_code == 200
            assert e.store.memory('api:test') == '' and e.store.get('api:test') == []
            assert e.store.memory('api:other') == 'other fact'
    asyncio.run(run())


def test_copywriter_wire_request_is_generation_without_source_text(tmp_path):
    e = engine(tmp_path)
    asyncio.run(e.generate('a', 'write a story'))
    request = e.client.invoke.call_args.args[0]
    assert e.client.invoke.call_args.kwargs == {'retries': 1, 'timeout': 55}
    wire = request.write()
    assert struct.unpack('<II', wire[:8]) == (0x8d7ae6af, 1 << 2)
    decoded = module.raw.core.TLObject.read(BytesIO(wire))
    assert decoded.text is None
    assert decoded.tone.custom_prompt == request.tone.custom_prompt
    result = ai_result('{"text":"answer","memory":null}')
    roundtrip = module.raw.core.TLObject.read(BytesIO(result.write()))
    assert module.copywriter_text(roundtrip) == '{"text":"answer","memory":null}'


def test_copywriter_rich_text_and_rejected_blocks():
    raw = module.raw.types
    result = ai_result('unused')
    result.result.blocks = [raw.PageBlockParagraph(text=raw.TextConcat(texts=[
        raw.TextPlain(text='{"text":'), raw.TextBold(text=raw.TextPlain(text='"answer"')),
        raw.TextEmpty(), raw.TextPlain(text=',"memory":null}')]))]
    assert module.copywriter_text(result) == '{"text":"answer","memory":null}'
    result.result.part = True
    with pytest.raises(module.InvalidReply):
        module.copywriter_text(result)
    result.result.part = False
    result.result.blocks = [raw.PageBlockUnsupported()]
    with pytest.raises(module.InvalidReply):
        module.copywriter_text(result)


def test_copywriter_prompt_limit_from_telegram():
    raw = module.raw.types
    config = raw.help.AppConfig(hash=1, config=raw.JsonObject(value=[
        raw.JsonObjectValue(key='aicompose_tone_prompt_length_max', value=raw.JsonNumber(value=4096.0))]))
    assert module.copywriter_prompt_limit(config) == 4096
    assert module.copywriter_prompt_limit(raw.help.AppConfigNotModified()) == 1024
    config.config.value[0].value = raw.JsonNumber(value=-1)
    with pytest.raises(RuntimeError):
        module.copywriter_prompt_limit(config)


def test_copywriter_small_limit_and_utf8(tmp_path):
    e = engine(tmp_path)
    e.prompt_limit = 1024
    asyncio.run(e.generate('a', 'Привет'))
    prompt = e.client.invoke.call_args.args[0].tone.custom_prompt
    assert module.text_length(prompt) <= 1024
    assert module.text_length('😀') == 4
    e.client.invoke.reset_mock()
    with pytest.raises(ValueError):
        asyncio.run(e.generate('a', '😀' * 500))
    e.client.invoke.assert_not_called()
    e.client.invoke.return_value = ai_result(json.dumps({'text': 'ok', 'memory': '😀' * 129}))
    with pytest.raises(module.InvalidReply):
        asyncio.run(e.generate('a', 'Запомни'))
    assert e.store.memory('a') == ''


@pytest.mark.parametrize('guest', [True, False])
def test_rich_reply_preserves_markdown_long_text_and_memory(tmp_path, guest):
    markdown = '# Заголовок 😀\n\n**Жирный**, *курсив*, [ссылка](https://example.com).\n\n- Один\n- Два\n\n> Цитата\n\n```python\nprint("<>&")\n```\n\n' + 'Длинный текст. ' * 400
    async def run():
        e = engine(tmp_path)
        e.client.invoke.return_value = ai_result(json.dumps({'text': markdown, 'memory': 'Пишет на Python.'}))
        bot = module.Bot('fake', e, {42})
        bot.me = {'id': 7, 'username': 'testbot'}
        bot.call = AsyncMock()
        message = {'message_id': 3, 'chat': {'id': 8, 'type': 'group'}, 'message_thread_id': 5,
                   'from': {'id': 42}, 'text': '@testbot Напиши статью'}
        if guest:
            message['guest_query_id'] = 'g1'
        await bot.handle({'guest_message' if guest else 'message': message})
        assert bot.call.await_count == 1
        args, data = bot.call.call_args
        if guest:
            assert args == ('answerGuestQuery',)
            assert data['guest_query_id'] == 'g1'
            content = data['result']['input_message_content']
            assert 'message_text' not in content
        else:
            assert args == ('sendRichMessage',)
            assert data['chat_id'] == 8 and data['message_thread_id'] == 5
            assert data['reply_parameters'] == {'message_id': 3}
            assert 'text' not in data
            content = data
        assert content['rich_message'] == {'markdown': markdown}
        key = f'tg:{"guest" if guest else "member"}:8:5:42'
        assert e.store.get(key)[-1]['content'] == markdown
        assert e.store.memory(key) == 'Пишет на Python.'
        await bot.http.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('guest', [True, False])
def test_rich_generation_error_stays_plain_text(tmp_path, guest):
    async def run():
        e = engine(tmp_path)
        e.client.invoke.return_value = ai_result('broken JSON')
        bot = module.Bot('fake', e, {42})
        bot.me = {'id': 7, 'username': 'testbot'}
        bot.call = AsyncMock()
        message = {'message_id': 3, 'chat': {'id': 8, 'type': 'private'}, 'from': {'id': 42}, 'text': 'hi'}
        if guest:
            message['guest_query_id'] = 'g1'
        await bot.handle({'guest_message' if guest else 'message': message})
        data = bot.call.call_args.kwargs
        if guest:
            content = data['result']['input_message_content']
            assert 'rich_message' not in content
            assert 'память не изменена' in content['message_text']
        else:
            assert bot.call.call_args.args == ('sendMessage',)
            assert 'rich_message' not in data
            assert 'память не изменена' in data['text']
        await bot.http.aclose()
    asyncio.run(run())


def test_rich_reply_length_limit_preserves_existing_data(tmp_path):
    e = engine(tmp_path)
    e.store.put('a', [], 'old memory')
    e.client.invoke.return_value = ai_result(json.dumps({'text': 'x' * 32768, 'memory': None}))
    assert len(asyncio.run(e.generate('a', 'write'))) == 32768
    old_turns = e.store.get('a')
    e.client.invoke.return_value = ai_result(json.dumps({'text': 'x' * 32769, 'memory': 'new memory'}))
    with pytest.raises(module.InvalidReply):
        asyncio.run(e.generate('a', 'write again'))
    assert e.store.get('a') == old_turns
    assert e.store.memory('a') == 'old memory'


def test_native_copywriter_article_preserves_formatting_without_writing_memory(tmp_path):
    raw = module.raw.types
    e = engine(tmp_path)
    e.store.put('a', [], 'old memory')
    result = ai_result('unused')
    result.result.blocks = [
        raw.PageBlockTitle(text=raw.TextPlain(text='Привет')),
        raw.PageBlockParagraph(text=raw.TextConcat(texts=[raw.TextBold(text=raw.TextPlain(text='Важно')),
            raw.TextPlain(text=': '), raw.TextUrl(text=raw.TextPlain(text='ссылка'), url='https://example.com/a(b)', webpage_id=0)])),
        raw.PageBlockList(items=[raw.PageListItemText(text=raw.TextPlain(text='Пункт'))]),
        raw.PageBlockBlockquote(text=raw.TextPlain(text='Цитата'), caption=raw.TextEmpty()),
        raw.PageBlockPreformatted(text=raw.TextPlain(text='print("hi")'), language='python'),
    ]
    e.client.invoke.return_value = result
    answer = asyncio.run(e.generate('a', 'Привет'))
    assert '# Привет' in answer and '**Важно**' in answer
    assert '[ссылка](https://example.com/a%28b%29)' in answer
    assert '- Пункт' in answer and '> Цитата' in answer
    assert '```python\nprint("hi")\n```' in answer
    assert e.store.memory('a') == 'old memory'
    assert e.store.get('a')[-1]['content'] == answer


def test_json_envelope_with_native_heading_is_recognized():
    result = ai_result('{"text":"Привет","memory":"new"}')
    result.result.blocks.insert(0, module.raw.types.PageBlockTitle(text=module.raw.types.TextPlain(text='Ответ')))
    reply = module.copywriter_reply(result)
    assert reply.text == 'Привет' and reply.memory == 'new'
    result.result.blocks.append(result.result.blocks[-1])
    with pytest.raises(module.InvalidReply):
        module.copywriter_reply(result)


def test_native_plain_reply_and_code_fence_escaping():
    raw = module.raw.types
    result = ai_result('unused')
    result.result.blocks = [raw.PageBlockParagraph(text=raw.TextPlain(text='Привет!'))]
    reply = module.copywriter_reply(result)
    assert reply.text == 'Привет\\!' and reply.memory is None
    result.result.blocks = [raw.PageBlockPreformatted(text=raw.TextPlain(text='```text\nhello\n```'), language='markdown')]
    assert module.copywriter_reply(result).text == '````markdown\n```text\nhello\n```\n````'


def test_bot_reports_timeout_without_writing_memory(tmp_path):
    async def run():
        e = engine(tmp_path)
        e.client.invoke.side_effect = TimeoutError()
        e.store.put('tg:member:8:0:42', [], 'old memory')
        bot = module.Bot('fake', e, {42})
        bot.me = {'id': 7, 'username': 'testbot'}
        bot.call = AsyncMock()
        await bot.handle({'message': {'message_id': 3, 'chat': {'id': 8, 'type': 'private'}, 'from': {'id': 42}, 'text': 'hi'}})
        assert 'не ответил за минуту' in bot.call.call_args.kwargs['text']
        assert e.store.memory('tg:member:8:0:42') == 'old memory'
        assert e.store.get('tg:member:8:0:42') == []
        await bot.http.aclose()
    asyncio.run(run())


def test_cyrillic_history_fits_wire_byte_limit_without_deleting_stored_turns(tmp_path):
    e = engine(tmp_path)
    e.prompt_limit = 1024
    turns = [{'role': 'user', 'content': 'Расскажи о погоде. ' * 10},
             {'role': 'assistant', 'content': 'Сегодня хороший день. ' * 10}]
    e.store.put('a', turns, 'Предпочитает русский язык.')
    asyncio.run(e.generate('a', '2+2'))
    prompt = e.client.invoke.call_args.args[0].tone.custom_prompt
    assert len(prompt.encode('utf-8')) <= 1024
    sent = json.loads(prompt.removeprefix(module.INSTRUCTION))
    assert sent['messages'] == [{'role': 'user', 'content': '2+2'}]
    assert sent['memory'] == 'Предпочитает русский язык.'
    assert e.store.get('a')[:2] == turns
    assert e.store.memory('a') == 'Предпочитает русский язык.'


def test_cyrillic_payload_checks_bytes_not_utf16():
    user = {'role': 'user', 'content': 'я' * 300}
    large = module.payload('system', [user], limit=12000)
    assert len(large.encode('utf-16-le')) // 2 < 1024
    assert len(large.encode('utf-8')) > 1024
    with pytest.raises(ValueError, match='байт UTF-8'):
        module.payload('system', [user], limit=1024)


def test_rpc_code_is_specific_and_does_not_expose_arbitrary_error_text():
    error = module.RPCError(raw_error=module.raw.types.RpcError(error_code=400, error_message='AICOMPOSE_TONE_PROMPT_INVALID'))
    assert module.rpc_error_code(error) == 'AICOMPOSE_TONE_PROMPT_INVALID'
    error.raw.error_message = 'Error includes private text or credentials'
    assert module.rpc_error_code(error) == 'RPCError'


def test_context_disabled_sends_only_prompt_and_preserves_saved_data(tmp_path):
    e = engine(tmp_path)
    e.use_context = False
    turns = [{'role': 'user', 'content': 'old request'}, {'role': 'assistant', 'content': 'old answer'}]
    e.store.put('a', turns, 'saved private fact')
    e.client.invoke.return_value.result.blocks = [module.raw.types.PageBlockParagraph(text=module.raw.types.TextPlain(text='4'))]
    assert asyncio.run(e.generate('a', '2+2', system='system must not be sent')) == '4'
    assert e.client.invoke.call_args.args[0].tone.custom_prompt == '2+2'
    assert e.store.get('a')[:2] == turns
    assert e.store.memory('a') == 'saved private fact'


def test_context_disabled_treats_json_as_content_not_memory_instructions(tmp_path):
    e = engine(tmp_path)
    e.use_context = False
    e.store.put('a', [], 'keep this')
    content = '{"text":"example","memory":"must not be saved"}'
    e.client.invoke.return_value = ai_result(content)
    assert asyncio.run(e.generate('a', 'Show JSON')) == '```json\n' + content + '\n```'
    assert e.store.memory('a') == 'keep this'


@pytest.mark.parametrize('guest', [True, False])
def test_context_disabled_does_not_append_reply_quote(tmp_path, guest):
    async def run():
        e = engine(tmp_path)
        e.use_context = False
        e.client.invoke.return_value.result.blocks = [module.raw.types.PageBlockParagraph(text=module.raw.types.TextPlain(text='4'))]
        bot = module.Bot('fake', e, {42})
        bot.me = {'id': 7, 'username': 'testbot'}
        bot.call = AsyncMock()
        message = {'message_id': 3, 'chat': {'id': 8, 'type': 'group'}, 'from': {'id': 42},
                   'text': '@testbot 2+2', 'reply_to_message': {'from': {'id': 7}, 'text': 'Do not send this quote'}}
        if guest: message['guest_query_id'] = 'g1'
        await bot.handle({'guest_message' if guest else 'message': message})
        assert e.client.invoke.call_args.args[0].tone.custom_prompt == '2+2'
        data = bot.call.call_args.kwargs
        content = data['result']['input_message_content'] if guest else data
        assert content['rich_message'] == {'markdown': '4'}
        await bot.http.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('prompt', [' ', 'я' * 513])
def test_context_disabled_still_validates_prompt(tmp_path, prompt):
    e = engine(tmp_path)
    e.use_context = False
    e.prompt_limit = 1024
    with pytest.raises(ValueError):
        asyncio.run(e.generate('a', prompt))
    e.client.invoke.assert_not_called()
    assert e.store.get('a') == []


def test_native_math_headings_details_and_table():
    raw = module.raw.types
    result = ai_result('unused')
    result.result.blocks = [
        raw.PageBlockHeading1(text=raw.TextPlain(text='Ответ')),
        raw.PageBlockHeading2(text=raw.TextPlain(text='Сумма')),
        raw.PageBlockMath(source='2 + 2 = 4'),
        raw.PageBlockParagraph(text=raw.TextMath(source='3 < 4')),
        raw.PageBlockDetails(title=raw.TextPlain(text='Пояснение'), blocks=[
            raw.PageBlockParagraph(text=raw.TextPlain(text='Текст <b>без HTML</b>'))]),
        raw.PageBlockTable(title=raw.TextEmpty(), rows=[
            raw.PageTableRow(cells=[raw.PageTableCell(header=True, text=raw.TextPlain(text='A'))]),
            raw.PageTableRow(cells=[raw.PageTableCell(text=raw.TextBold(text=raw.TextPlain(text='4')))])]),
    ]
    answer = module.copywriter_reply(result, structured=False).text
    assert '# Ответ' in answer and '## Сумма' in answer
    assert '<tg-math-block>2 + 2 = 4</tg-math-block>' in answer
    assert '<tg-math>3 &lt; 4</tg-math>' in answer
    assert '<details><summary>Пояснение</summary>' in answer
    assert '&lt;b&gt;без HTML&lt;/b&gt;' in answer
    assert '| A |\n| --- |\n| **4** |' in answer


def test_nested_native_quote_and_thinking():
    raw = module.raw.types
    answer = module.rich_markdown([
        raw.PageBlockBlockquoteBlocks(blocks=[raw.PageBlockParagraph(text=raw.TextPlain(text='Цитата'))], caption=raw.TextEmpty()),
        raw.PageBlockThinking(text=raw.TextPlain(text='Примечание'))])
    assert '> Цитата' in answer
    assert '<details><summary>Детали ответа</summary>' in answer
    assert '<tg-thinking>' not in answer
