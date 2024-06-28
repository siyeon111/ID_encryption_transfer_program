import mysql.connector
from mysql.connector import IntegrityError

import os
import mySQL_def as SQL


def read_file(file_path):
    with open(file_path, "rb") as file:
        return file.read()


def send_user_info(id):
    encryption_directory = os.getcwd() + "/encryption_file"
    user_id = id  # 예시 ID, 실제 적용 시 수정 필요

    # 파일 이름 설정
    files = {
        "img_iv": id + "_jpg_iv.bin",
        "txt_iv": id + "_txt_iv.bin",
        "img_key": id + "_jpg_key.bin",
        "txt_key": id + "_txt_key.bin",
        "img_bin": id + "_jpg_encrypted.bin",
        "txt_bin": id + "_txt_encrypted.bin",
    }

    # 파일 내용 읽기
    file_contents = {
        name: read_file(os.path.join(encryption_directory, filename))
        for name, filename in files.items()
    }

    # 데이터베이스 연결 및 데이터 삽입
    db_conn = SQL.MySQLConnector("192.168.0.62", "user1", "1234", "identification")
    db_conn.connect()

    result = db_conn.check_id_exists(id)
    if result == True:
        db_conn.delete_info("user_info", id)
    db_conn.insert_into_user_info(
        user_id,
        file_contents["img_iv"],
        file_contents["txt_iv"],
        file_contents["img_key"],
        file_contents["txt_key"],
        file_contents["img_bin"],
        file_contents["txt_bin"],
    )
    db_conn.disconnect()
