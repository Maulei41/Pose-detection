import base64

import streamlit as st


class Add_html(object):
    def __init__(self):
        a=1
    def autoplay_audio(self, file_path: str):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            return md
    def popup_window(self):
        popup = f"""
                """
        return popup
