import mysql.connector
from mysql.connector import IntegrityError
from mysql.connector import Error
import os


class MySQLConnector:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )
        self.cursor = self.conn.cursor()

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def delete_info(
        self, table_name, id_value
    ):  # 테이블에서 한 행(ID 받아서) 정보 삭제하기
        self.table_name = table_name
        query = "DELETE FROM " + self.table_name + " WHERE id = %s"
        values = (id_value,)
        self.cursor.execute(query, values)
        self.conn.commit()
        print(f"{self.cursor.rowcount} row(s) deleted from company_info")

    def count_table(self, table_name):  # 데이터(테이블 안 정보[행]) 개수 세기
        self.table_name = table_name
        query = "SELECT COUNT(*) FROM " + self.table_name
        self.cursor.execute(query)
        row_count = self.cursor.fetchone()[0]
        print("정보 수 : ", row_count)

    # def insert_into_id_pwd(  # id_pwd 테이블에 정보 대입 함수
    #     self, id, password, phone_number
    # ):
    #     query = "INSERT INTO id_pwd (id, password, phone_number) VALUES (%s, %s, %s)"
    #     values = (id, password, phone_number)
    #     self.cursor.execute(query, values)
    #     self.conn.commit()
    #     print(f"Added to id_pwd: {self.cursor.rowcount} row(s)")
    #     return id

    # def insert_into_id_pwd(self, id, password, phone_number):
    #     try:
    #         query = (
    #             "INSERT INTO id_pwd (id, password, phone_number) VALUES (%s, %s, %s)"
    #         )
    #         values = (id, password, phone_number)
    #         self.cursor.execute(query, values)
    #         self.conn.commit()
    #         print(f"Added to id_pwd: {self.cursor.rowcount} row(s)")
    #     except IntegrityError:
    #         print("이미 존재하는 id입니다.")
    #         self.conn.rollback()  # 에러 발생 시, 데이터베이스 상태를 이전 상태로 되돌림
    #         return None
    #     return id
    def insert_into_id_pwd(self, id, password, phone_number):
        try:
            query = (
                "INSERT INTO id_pwd (id, password, phone_number) VALUES (%s, %s, %s)"
            )
            values = (id, password, phone_number)
            self.cursor.execute(query, values)
            self.conn.commit()
            print(f"Added to id_pwd: {self.cursor.rowcount} row(s)")
        except IntegrityError:
            print("이미 존재하는 id입니다.")
            self.conn.rollback()  # 에러 발생 시, 데이터베이스 상태를 이전 상태로 되돌림
            return None
        return id

    # def insert_into_user_info(
    #     self, user_id, img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin
    # ):
    #     query = """
    #     INSERT INTO user_info
    #     (id, img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin)
    #     VALUES (%s, %s, %s, %s, %s, %s, %s)
    #     """
    #     values = (user_id, img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin)
    #     self.cursor.execute(query, values)
    #     self.conn.commit()
    #     print(f"Data inserted for user {user_id} into user_info.")

    def insert_into_user_info(
        self, id, img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin
    ):
        try:
            # query = "INSERT INTO user_info (id, img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            query = (
                "INSERT INTO user_info (id, img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "img_iv=VALUES(img_iv), txt_iv=VALUES(txt_iv), "
                "img_key=VALUES(img_key), txt_key=VALUES(txt_key), "
                "img_bin=VALUES(img_bin), txt_bin=VALUES(txt_bin)"
            )
            values = (id, img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin)
            self.cursor.execute(query, values)
            self.conn.commit()
            print(f"Added to user_info: {self.cursor.rowcount} row(s)")
        except IntegrityError as e:
            if "foreign key constraint fails" in str(e):
                print("id_pwd 테이블에 해당 id가 존재하지 않습니다.")
            elif "Duplicate entry" in str(e):
                print("이미 존재하는 id입니다.")
            else:
                print("데이터베이스 오류가 발생했습니다.")
            self.conn.rollback()  # 에러 발생 시, 데이터베이스 상태를 이전 상태로 되돌림

    def find_in_id_pwd(self, id):
        query = "SELECT password, phone_number FROM user_info WHERE id = %s"
        self.cursor.execute(query, (id,))
        result = self.cursor.fetchone()
        if result:
            password, phone_number = result
            print("ID : %s" % id)
            print("Password : %s" % password)
            print("phone_number : %s" % phone_number)
            return id, password, phone_number
        else:
            print("ID에 해당하는 데이터를 찾을 수 없습니다.")
            return None

    # def find_in_user_info(
    #     self, id
    # ):  # user_info 테이블에서 정보(iv, key, 이미지 텍스트 바이너리 내용) 다 가져오기
    #     # user_info 테이블에서 데이터 조회
    #     query = "SELECT img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin FROM user_info WHERE id = %s"
    #     self.cursor.execute(query, (id,))
    #     result = self.cursor.fetchone()

    #     if result:
    #         # 변수로 저장
    #         img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin = result

    #         # img_bin 데이터를 image_data.bin 파일로 저장
    #         with open(
    #             "image_data.bin", "wb"
    #         ) as img_file:  ##################################################
    #             img_file.write(img_bin)

    #         # txt_bin 데이터를 text_data.bin 파일로 저장
    #         with open("text_data.bin", "wb") as txt_file:
    #             txt_file.write(txt_bin)

    #         print("Files saved: image_data.bin, text_data.bin")
    #         print(
    #             "img_iv : %s\ntxt_iv : %s\nimg_key : %s\ntxt_key : %s"
    #             % (img_iv, txt_iv, img_key, txt_key)
    #         )
    #         return img_iv, txt_iv, img_key, txt_key
    #     else:
    #         print("No data found with the given ID")
    #         return None

    # 광욱이가 수정해본거
    def find_in_user_info(
        self, id
    ):  # user_info 테이블에서 정보(iv, key, 이미지 텍스트 바이너리 내용) 다 가져오기
        # user_info 테이블에서 데이터 조회
        query = "SELECT img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin FROM user_info WHERE id = %s"
        self.cursor.execute(query, (id,))
        result = self.cursor.fetchone()

        if result:
            # 변수로 저장
            img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin = result

            print(
                "img_iv : %s\ntxt_iv : %s\nimg_key : %s\ntxt_key : %s\nimg : 성공\ntxt : 성공"
                % (img_iv, txt_iv, img_key, txt_key)
            )
            return img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin
        else:
            print("No data found with the given ID")
            return None

    # def find_in_user_info(self, id):
    #     query = "SELECT img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin FROM user_info WHERE id = %s"
    #     self.cursor.execute(query, (id,))
    #     result = self.cursor.fetchone()

    #     if result:
    #         img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin = result
    #         return img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin
    #     else:
    #         print("No data found with the given ID")
    #         return None

    #     def check_id_exists(self, id):
    #         try:
    #             query = "SELECT EXISTS(SELECT * FROM id_pwd WHERE id = %s)"
    #             self.cursor.execute(query, (id,))
    #             exists = self.cursor.fetchone()[0]
    #             return not exists
    #         except Error as e:
    #             print(f"Error: {e}")
    #             return False  # 에러 발생 시 False 반환

    # def use_check_id_exists(id):
    #     db_conn = MySQLConnector("192.168.0.62", "user1", "1234", "identification")
    #     db_conn.connect()
    #     db_conn.check_id_exists(id)
    #     db_conn.disconnect()

    def check_id_exists(self, id):
        try:
            query = "SELECT EXISTS(SELECT * FROM id_pwd WHERE id = %s)"
            self.cursor.execute(query, (id,))
            exists = self.cursor.fetchone()[0]
            return exists  # 존재하면 True, 존재하지 않으면 False 반환
        except Error as e:
            print(f"Error: {e}")
            return False  # 에러 발생 시 False 반환


def use_check_id_exists(id):
    db_conn = MySQLConnector("192.168.0.62", "user1", "1234", "identification")
    db_conn.connect()
    result = db_conn.check_id_exists(id)  # 결과를 받아옵니다.
    db_conn.disconnect()
    if result:
        return True
    else:
        return False  # 결과를 반환합니다.
