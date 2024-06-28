# import crypto
import sys

# sys.modules["Crypto"] = crypto
from mySQL_def import MySQLConnector  # mySQL_def.py에서 MySQLConnector 클래스를 가져옴
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import os


# 복호화 함수
def decrypt_data(encrypted_data, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    original_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
    return original_data


# 복호화된 데이터 저장 함수
def save_decrypted(file_path, data):
    with open(file_path, "wb") as file:
        file.write(data)


# 메인 복호화 로직
def decrypt_user_data(user_id, decrypted_directory):
    # 데이터베이스 객체 생성 및 연결
    db_conn = MySQLConnector("192.168.0.62", "user1", "1234", "identification")
    db_conn.connect()

    # user_info 테이블에서 해당 사용자의 데이터 가져오기
    img_iv, txt_iv, img_key, txt_key, img_bin, txt_bin = db_conn.find_in_user_info(
        user_id
    )  # 이 줄을 수정했습니다.

    # 복호화된 데이터 저장할 디렉토리가 없으면 생성
    if not os.path.exists(decrypted_directory):
        os.makedirs(decrypted_directory)

    # 이미지 데이터 복호화 및 저장
    image_file_path = os.path.join(decrypted_directory, f"{user_id}.jpg")
    decrypted_img = decrypt_data(img_bin, img_key, img_iv)
    save_decrypted(image_file_path, decrypted_img)

    # 텍스트 데이터 복호화 및 저장
    text_file_path = os.path.join(decrypted_directory, f"{user_id}.txt")
    decrypted_txt = decrypt_data(txt_bin, txt_key, txt_iv)
    save_decrypted(text_file_path, decrypted_txt)

    print("Decryption completed for user:", user_id)

    # 데이터베이스 연결 종료
    db_conn.disconnect()


# 사용자 ID와 저장할 디렉토리 지정
# user_id = "iii"  # 실제 user_id로 대체 필요
# decrypted_directory = "decrypted_file"

# decrypt_user_data(user_id, decrypted_directory)
