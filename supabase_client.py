from supabase import create_client
import streamlit as st

_BUCKET = "gpx_routes"


def _client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def upload_gpx(xml_str: str, filename: str) -> tuple[bool, str]:
    """GPXファイルをSupabase Storageにアップロードする。
    Returns: (成功フラグ, 公開URL or エラーメッセージ)
    """
    try:
        sb = _client()
        sb.storage.from_(_BUCKET).upload(
            path=f"{filename}.gpx",
            file=xml_str.encode("utf-8"),
            file_options={"content-type": "application/gpx+xml", "upsert": "true"},
        )
        public_url = sb.storage.from_(_BUCKET).get_public_url(f"{filename}.gpx")
        return True, public_url
    except Exception as e:
        return False, str(e)
