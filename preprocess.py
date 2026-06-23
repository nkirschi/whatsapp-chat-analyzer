import argparse
import datetime
import pickle as pkl
from collections import Counter

import polars as pl
import regex as re

# iOS export: [dd.mm.yy, hh:mm:ss] author: message
IOS_HEADER = re.compile(r"^\[\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2}\] ")
# Android export: dd/mm/yy, hh:mm - author: message
ANDROID_HEADER = re.compile(
    r"^(\d{1,2}[./]\d{1,2}[./]\d{2,4}, "
    r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?) - (.*)$"
)


def preprocess(input_file, include_metaAI=False):
    with open(input_file, "r", encoding="utf-8") as f:
        chat_str = f.read()

    if _detect_format(chat_str) == "android":
        return _preprocess_android(chat_str, include_metaAI)
    return _preprocess_ios(chat_str, include_metaAI)


def _detect_format(chat_str):
    """Inspect the first lines to decide between the iOS and Android export format."""
    for line in chat_str.splitlines():
        if not line.strip():
            continue
        if IOS_HEADER.match(line):
            return "ios"
        if ANDROID_HEADER.match(line):
            return "android"
    raise ValueError(
        "Could not recognize the chat format. Expected an iOS export "
        "('[dd.mm.yy, hh:mm:ss] author: message') or an Android export "
        "('dd/mm/yy, hh:mm - author: message')."
    )


def _preprocess_ios(chat_str, include_metaAI=False):
    info = {}

    # Count media types and call time
    media_counts = Counter()
    call_time = Counter()
    media_pattern = r"^\u200E(.*?)\u200E(\w+)\s\w+$"
    call_pattern = r": \u200E(\w+)\. \u200E\u200E(\d+)\s(\w+)\.\s\u2022.*$"

    def replace_media(match):
        key = match.group(2)
        media_counts[key] += 1
        return f"{match.group(1)}<{key}>"

    def replace_call(match):
        key = match.group(1)
        call_time[match.group(3)] += int(match.group(2))
        media_counts[key] += 1
        return f": <{key}>"

    chat_str = re.sub(media_pattern, replace_media, chat_str, flags=re.MULTILINE)
    chat_str = re.sub(call_pattern, replace_call, chat_str, flags=re.MULTILINE)

    total_time_called = int(
        datetime.timedelta(
            hours=call_time.get("Std", 0),
            minutes=call_time.get("Min", 0),
            seconds=call_time.get("Sek", 0),
        ).total_seconds()
    )
    H = total_time_called // 3600
    M = (total_time_called % 3600) // 60

    # remove LTR/RTL markers
    chat_str = re.sub(r"[\u2066-\u2069]", "", chat_str)
    chat_str = re.sub(r"[\p{Cf}]", "", chat_str)

    # [dd.mm.yy, hh:mm:ss] author: message
    message_mask = (
        r"\[(\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2})\] (.*?): ?(.*?)(?=\n\[|$)"
    )
    messages = re.findall(message_mask, chat_str, re.DOTALL)

    df = (
        pl.DataFrame(messages, schema=["datetime", "author", "message"], orient="row")
        .with_columns(
            datetime=pl.col("datetime").str.strptime(pl.Datetime, "%d.%m.%y, %H:%M:%S"),
            message_length=pl.col("message").str.len_chars(),
        )
        .sort("datetime")
    )

    # filter out: First author = chatname, Meta AI messages if flag is not set
    # either group-> multiple authors + metaai, first message is group name
    # or one-on-one chat -> 2 authors + metaai, first author is other person name

    chat_name = df["author"].to_list()[0]
    author_unique = set(df["author"].unique().to_list()) - {"Meta AI"}

    # if authors without meta ai >2 -> group chat
    if len(author_unique) > 2:
        is_group = True
    else:
        is_group = False

    author_filter = []
    if is_group:
        author_filter.append(chat_name)
    if not include_metaAI:
        author_filter.append("Meta AI")
    df = df.filter(~pl.col("author").is_in(author_filter))

    info["chat_name"] = chat_name
    info["is_group"] = is_group
    info["df"] = df
    info["media_counts"] = media_counts
    info["total_call_time"] = {"h": H, "m": M}

    return info


# Android media placeholders. Generic "<Media omitted>" exports carry no type
# information; some locales/versions use typed "<type> omitted" placeholders.
ANDROID_TYPED_MEDIA = re.compile(
    r"^(image|video|audio|GIF|sticker|document|Contact card) omitted$"
)


def _classify_android_media(message):
    """Return a media-type label if the message is a media placeholder, else None."""
    msg = message.strip()
    if msg == "<Media omitted>":
        return "Media"
    m = ANDROID_TYPED_MEDIA.match(msg)
    if m:
        return m.group(1).capitalize()
    return None


def _parse_android_datetime(df):
    """Parse the string 'datetime' column, trying the common Android layouts."""
    candidate_formats = [
        "%d/%m/%y, %H:%M",
        "%d/%m/%Y, %H:%M",
        "%d.%m.%y, %H:%M",
        "%d.%m.%Y, %H:%M",
        "%m/%d/%y, %H:%M",
        "%m/%d/%Y, %H:%M",
        "%d/%m/%y, %I:%M %p",
        "%m/%d/%y, %I:%M %p",
        "%d/%m/%y, %H:%M:%S",
        "%d/%m/%Y, %H:%M:%S",
    ]
    best_fmt, best_parsed = None, -1
    for fmt in candidate_formats:
        parsed = df["datetime"].str.strptime(pl.Datetime, fmt, strict=False)
        n_ok = parsed.is_not_null().sum()
        if n_ok > best_parsed:
            best_fmt, best_parsed = fmt, n_ok
    if best_parsed <= 0:
        raise ValueError(
            "Could not parse any Android timestamps "
            f"(example: {df['datetime'][0]!r})."
        )
    return df.with_columns(
        datetime=pl.col("datetime").str.strptime(pl.Datetime, best_fmt, strict=False)
    ).drop_nulls("datetime")


def _preprocess_android(chat_str, include_metaAI=False):
    info = {}

    # Normalize non-breaking / directional spaces, strip LTR/RTL & format markers
    chat_str = chat_str.replace("\u00a0", " ").replace("\u202f", " ")
    chat_str = re.sub(r"[\p{Cf}]", "", chat_str)

    # Group consecutive lines into entries; lines without a timestamp header are
    # continuations of the previous (multi-line) message.
    entries = []  # [datetime_str, body]
    for line in chat_str.splitlines():
        m = ANDROID_HEADER.match(line)
        if m:
            entries.append([m.group(1), m.group(2)])
        elif entries:
            entries[-1][1] += "\n" + line

    # Split each entry into author/message; entries without "author: " are
    # system notices (encryption notice, group changes, security code, ...).
    author_mask = re.compile(r"^([^:\n]{1,100}?): ?(.*)$", re.DOTALL)
    rows = []
    media_counts = Counter()
    system_messages = []
    for dt_str, body in entries:
        am = author_mask.match(body)
        if not am:
            system_messages.append(body)
            continue
        author, message = am.group(1).strip(), am.group(2)
        rows.append((dt_str, author, message))
        media_type = _classify_android_media(message)
        if media_type is not None:
            media_counts[media_type] += 1

    if not rows:
        raise ValueError("No messages were parsed from the Android chat export.")

    df = pl.DataFrame(rows, schema=["datetime", "author", "message"], orient="row")
    df = _parse_android_datetime(df)
    df = df.with_columns(
        message_length=pl.col("message").str.len_chars(),
    ).sort("datetime")

    author_unique = set(df["author"].unique().to_list()) - {"Meta AI"}
    is_group = len(author_unique) > 2

    # Android exports don't carry a group-name pseudo-author, so derive the chat
    # name from system notices for groups; use the first speaker otherwise.
    if is_group:
        chat_name = _extract_android_group_name(system_messages) or "Group Chat"
    else:
        chat_name = df["author"].to_list()[0]

    # Never drop a real author for Android; only optionally drop Meta AI.
    if not include_metaAI:
        df = df.filter(pl.col("author") != "Meta AI")

    # Android text exports don't include call durations.
    info["chat_name"] = chat_name
    info["is_group"] = is_group
    info["df"] = df
    info["media_counts"] = media_counts
    info["total_call_time"] = {"h": 0, "m": 0}

    return info


def _extract_android_group_name(system_messages):
    """Best-effort extraction of the group subject from system notices."""
    subject_re = re.compile(
        r'(?:created group|changed the subject (?:from .*? )?to) "(.+?)"'
    )
    for msg in system_messages:
        m = subject_re.search(msg)
        if m:
            return m.group(1)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess chat data")
    parser.add_argument("input_file", help="Path to the input chat file")
    parser.add_argument(
        "--include_metaAI", action="store_true", help="Include messages from Meta AI"
    )
    args = parser.parse_args()

    info = preprocess(args.input_file, args.include_metaAI)

    # export to pickle
    pkl.dump(
        info, open(f"{args.input_file.replace('.txt', '')}_preprocessed.pkl", "wb")
    )
