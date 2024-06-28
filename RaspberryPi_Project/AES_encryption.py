### iv도 저장 ################################################################################

# import crypto
import sys

# sys.modules["Crypto"] = crypto
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
import os


def encrypt_data(data, key):
    cipher = AES.new(
        key, AES.MODE_CBC
    )  # 새 AES 암호화 객체를 생성하고, CBC 모드를 사용합니다.
    ct_bytes = cipher.encrypt(
        pad(data, AES.block_size)
    )  # 데이터를 패딩하고 암호화합니다.
    iv = cipher.iv  # 암호화에 사용된 초기화 벡터(IV)를 가져옵니다.
    return iv, ct_bytes  # 초기화 벡터(IV)와 암호화된 데이터를 반환합니다.


def save_encrypted(file_path, data):
    with open(file_path, "wb") as file:
        file.write(data)  # 암호화된 데이터만 파일에 씁니다.


def save_iv(iv, iv_file_path):
    with open(iv_file_path, "wb") as iv_file:
        iv_file.write(iv)  # IV를 별도의 파일에 저장합니다.


def save_key(key, key_file_path):
    with open(key_file_path, "wb") as key_file:
        key_file.write(key)  # 암호화 키를 파일에 저장합니다.


def make_encrypted_file(original_file_name):
    original_file_path = os.getcwd() + "/information_file"
    if not os.path.exists(original_file_path):
        os.makedirs(original_file_path)

    encryption_file_path = os.getcwd() + "/encryption_file"
    original_file_path = os.path.join(original_file_path, original_file_name)

    # original_file_name = os.path.splitext(os.path.basename(original_file_path))[0]
    # 파일의 경로에서 파일명만을 추출하고, 그 파일명에서 확장자를 제거하는 역할
    original_file_name = original_file_name.replace(".", "_")

    encrypted_file_name = f"{original_file_name}_encrypted.bin"
    key_file_name = f"{original_file_name}_key.bin"
    iv_file_name = f"{original_file_name}_iv.bin"

    encrypted_file_path = os.path.join(encryption_file_path, encrypted_file_name)
    key_file_path = os.path.join(encryption_file_path, key_file_name)
    iv_file_path = os.path.join(encryption_file_path, iv_file_name)

    key = get_random_bytes(16)  # 암호화에 사용될 키를 무작위로 생성합니다.
    save_key(key, key_file_path)  # 키를 파일에 저장합니다.

    with open(original_file_path, "rb") as file:
        original_data = file.read()  # 원본 데이터를 읽습니다.

    iv, encrypted_data = encrypt_data(original_data, key)  # 데이터를 암호화합니다.
    save_encrypted(
        encrypted_file_path, encrypted_data
    )  # 암호화된 데이터를 파일에 저장합니다.
    save_iv(iv, iv_file_path)  # IV를 별도의 파일에 저장합니다.

    print(f"Encrypted file saved to: {encrypted_file_path}")
    print(f"Key file saved to: {key_file_path}")
    print(f"IV file saved to: {iv_file_path}")


# # # 실행 예시
# make_encrypted_file("iii.jpg")

# text_path = "C:\\LeeJunYoung\\Python_Project\\information_file\\user1_name.txt"
# make_encrypted_file(text_path)
