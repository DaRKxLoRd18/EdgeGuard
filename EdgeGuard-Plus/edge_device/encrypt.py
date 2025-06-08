# Placeholder for encrypt.py
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import base64
import os
import json

class MetadataEncryptor:
    def __init__(self, key=None):
        self.key = key or os.urandom(32)  # 256-bit key
    
    def encrypt(self, data_dict):
        iv = os.urandom(16)
        data = json.dumps(data_dict).encode('utf-8')
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(data, AES.block_size))
        return {
            'iv': base64.b64encode(iv).decode(),
            'ciphertext': base64.b64encode(ciphertext).decode()
        }
