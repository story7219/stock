import os
import keyring
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = self._get_or_create_key()
        
    def _get_or_create_key(self):
        """암호화 키 생성 또는 가져오기"""
        key = keyring.get_password("trading_app", "encryption_key")
        if not key:
            key = Fernet.generate_key().decode()
            keyring.set_password("trading_app", "encryption_key", key)
        return key.encode()
    
    def encrypt_api_key(self, api_key: str) -> str:
        """API 키 암호화"""
        f = Fernet(self.encryption_key)
        return f.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """API 키 복호화"""
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_key.encode()).decode() 