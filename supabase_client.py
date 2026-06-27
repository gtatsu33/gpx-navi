from supabase import create_client
import streamlit as st

_BUCKET = "gpx_routes"


def _client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def upload_gpx(
    xml_str: str,
    filename: str,
    display_name: str,
    distance_m: float | None = None,
    elevation_gain_m: float | None = None,
) -> tuple[bool, str, str | None]:
    """GPXをStorageにアップロードしてroute_filesにメタデータを登録する。

    Returns: (成功フラグ, メッセージ, エラー種別)
    エラー種別: None=成功 / "file_key_dup" / "display_name_dup" / "other"
    """
    file_key = f"{filename}.gpx"
    sb = _client()

    try:
        sb.storage.from_(_BUCKET).upload(
            path=file_key,
            file=xml_str.encode("utf-8"),
            file_options={"content-type": "application/gpx+xml"},
        )
    except Exception as e:
        msg = str(e)
        if any(w in msg.lower() for w in ("already exists", "duplicate", "409")):
            return False, "このファイルキーは既に使用されています。別のファイル名を指定してください。", "file_key_dup"
        return False, f"Storageへのアップロードに失敗しました: {msg}", "other"

    try:
        sb.table("route_files").insert({
            "file_key": file_key,
            "display_name": display_name,
            "distance_m": distance_m,
            "elevation_gain_m": elevation_gain_m,
        }).execute()
    except Exception as e:
        # Storage をロールバック
        try:
            sb.storage.from_(_BUCKET).remove([file_key])
        except Exception:
            pass
        msg = str(e)
        if "route_files_display_name_key" in msg or (
            "23505" in msg and "display_name" in msg
        ):
            return False, "この表示名は既に使用されています。別のファイル名を入力してください。", "display_name_dup"
        if "23505" in msg:
            return False, "このファイルキーは既に使用されています。別のファイル名を指定してください。", "file_key_dup"
        return False, f"DB登録に失敗しました（Storageはロールバック済み）: {msg}", "other"

    return True, file_key, None


def list_routes() -> tuple[bool, list | str]:
    """route_files を新しい順で全件取得する。"""
    try:
        sb = _client()
        res = sb.table("route_files").select("*").order("created_at", desc=True).execute()
        return True, res.data
    except Exception as e:
        return False, str(e)


def download_gpx(file_key: str) -> tuple[bool, str]:
    """Storage から GPX をダウンロードして文字列で返す。"""
    try:
        sb = _client()
        data = sb.storage.from_(_BUCKET).download(file_key)
        return True, data.decode("utf-8")
    except Exception as e:
        return False, str(e)
