from app.domain.conversation import Conversation, Message, Role


def test_new_conversation_has_no_messages():
    conv = Conversation(session_id="s1")
    assert conv.messages == []


def test_append_user_records_role_and_content():
    conv = Conversation(session_id="s1")
    conv.append_user("hello")
    assert len(conv.messages) == 1
    assert conv.messages[0].role is Role.USER
    assert conv.messages[0].content == "hello"


def test_append_assistant_records_role_and_content():
    conv = Conversation(session_id="s1")
    conv.append_assistant("hi there")
    assert conv.messages[0].role is Role.ASSISTANT
    assert conv.messages[0].content == "hi there"


def test_append_returns_the_message_with_a_timestamp():
    conv = Conversation(session_id="s1")
    msg = conv.append_user("hello")
    assert isinstance(msg, Message)
    assert msg.created_at is not None


def test_messages_preserve_insertion_order():
    conv = Conversation(session_id="s1")
    conv.append_user("first")
    conv.append_assistant("second")
    conv.append_user("third")
    assert [m.content for m in conv.messages] == ["first", "second", "third"]


def test_to_chat_messages_returns_role_content_dicts_in_order():
    conv = Conversation(session_id="s1")
    conv.append_user("ping")
    conv.append_assistant("pong")
    assert conv.to_chat_messages() == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]


def test_history_caps_non_system_messages_at_max_history():
    conv = Conversation(session_id="s1", max_history=4)
    for i in range(10):
        conv.append_user(f"u{i}")
    assert len(conv.messages) == 4
    assert [m.content for m in conv.messages] == ["u6", "u7", "u8", "u9"]


def test_trimming_preserves_system_messages_even_when_over_cap():
    conv = Conversation(session_id="s1", max_history=2)
    conv.append(Role.SYSTEM, "you are FRIDAY")
    for i in range(5):
        conv.append_user(f"u{i}")
    contents = [m.content for m in conv.messages]
    assert contents[0] == "you are FRIDAY"
    assert contents[1:] == ["u3", "u4"]


def test_message_is_immutable():
    conv = Conversation(session_id="s1")
    msg = conv.append_user("hello")
    import dataclasses
    assert not dataclasses.is_dataclass(msg) or getattr(type(msg), "__dataclass_params__").frozen
