## iv 추가로 보내기 ####################################################################################################

import mysql.connector
from mysql.connector import IntegrityError

import os
import mySQL_def as SQL


def read_file(file_path):
    with open(file_path, "rb") as file:
        return file.read()


def send_id_pwd(id, password, phone_number):
    # encryption_directory = "encryption_file"

    # # 파일 이름 설정
    # files = {
    #     "id" : id,
    #     "password": password,
    #     "phone_number" : phone_number,
    # }

    # # 파일 내용 읽기
    # file_contents = {
    #     name: read_file(os.path.join(encryption_directory, filename))
    #     for name, filename in files.items()
    # }

    # 데이터베이스 연결 및 데이터 삽입
    db_conn = SQL.MySQLConnector("192.168.0.62", "user1", "1234", "identification")
    db_conn.connect()
    db_conn.insert_into_id_pwd(id, password, phone_number)
    db_conn.disconnect()
