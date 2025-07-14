#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: security_manager.py
모듈: 보안 관리 시스템
목적: API 키 관리, 암호화, 보안 검증

Author: Security Engineer
Created: 2025-07-10
Version: 1.0.0
"""

import os
import base64
import hashlib
import secrets
import logging
from typing import Dict
import Optional
import Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import json
from datetime import datetime
import timedelta

logger = logging.getLogger(__name__)

class SecurityManager:
    """보안 관리 시스템"""

    def __init__(self):
        self.logger = logger
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        self.api_keys = {}
        self.access_tokens = {}

    def _get_or_create_encryption_key(self) -> bytes:
        """암호화 키 생성 또는 로드"""
        key_file = ".encryption_key"

        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"암호화 키 로드 실패: {e}")

        # 새 키 생성
        key = Fernet.generate_key()
        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # 소유자만 읽기/쓰기
            self.logger.info("새 암호화 키 생성됨")
        except Exception as e:
            self.logger.error(f"암호화 키 저장 실패: {e}")

        return key

    def encrypt_data(self, data: str) -> str:
        """데이터 암호화"""
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"데이터 암호화 실패: {e}")
            return ""

    def decrypt_data(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"데이터 복호화 실패: {e}")
            return ""

    def hash_password(self, password: str) -> str:
        """패스워드 해싱"""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}:{hash_obj.hex()}"

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """패스워드 검증"""
        try:
            salt, hash_hex = hashed_password.split(':')
            hash_obj = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return hash_obj.hex() == hash_hex
        except Exception as e:
            self.logger.error(f"패스워드 검증 실패: {e}")
            return False

    def generate_api_key(self, service_name: str) -> str:
        """API 키 생성"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[service_name] = {
            'key': api_key,
            'created_at': datetime.now().isoformat(),
            'last_used': None
        }
        self.logger.info(f"API 키 생성됨: {service_name}")
        return api_key

    def validate_api_key(self, service_name: str, api_key: str) -> bool:
        """API 키 검증"""
        if service_name not in self.api_keys:
            return False

        stored_key = self.api_keys[service_name]['key']
        if stored_key == api_key:
            self.api_keys[service_name]['last_used'] = datetime.now().isoformat()
            return True

        return False

    def generate_access_token(self, user_id: str, expires_in: int = 3600) -> str:
        """액세스 토큰 생성"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        self.access_tokens[token] = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat()
        }

        self.logger.info(f"액세스 토큰 생성됨: {user_id}")
        return token

    def validate_access_token(self, token: str) -> Optional[str]:
        """액세스 토큰 검증"""
        if token not in self.access_tokens:
            return None

        token_data = self.access_tokens[token]
        expires_at = datetime.fromisoformat(token_data['expires_at'])

        if datetime.now() > expires_at:
            del self.access_tokens[token]
            return None

        return token_data['user_id']

    def sanitize_input(self, user_input: str) -> str:
        """사용자 입력 새니타이징"""
        # XSS 방지
        dangerous_chars = ['<', '>', '"', "'", '&', 'script', 'javascript']
        sanitized = user_input.lower()

        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')

        return sanitized.strip()

    def validate_file_path(self, file_path: str) -> bool:
        """파일 경로 검증"""
        # 경로 순회 공격 방지
        normalized_path = os.path.normpath(file_path)
        if '..' in normalized_path or normalized_path.startswith('/'):
            return False

        return True

    def encrypt_sensitive_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """민감한 설정 암호화"""
        encrypted_config = {}
        sensitive_keys = ['password', 'secret', 'key', 'token']

        for key, value in config_data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                encrypted_config[key] = self.encrypt_data(str(value))
            else:
                encrypted_config[key] = value

        return encrypted_config

    def decrypt_sensitive_config(self, encrypted_config: Dict[str, Any]) -> Dict[str, Any]:
        """암호화된 설정 복호화"""
        decrypted_config = {}
        sensitive_keys = ['password', 'secret', 'key', 'token']

        for key, value in encrypted_config.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                decrypted_config[key] = self.decrypt_data(str(value))
            else:
                decrypted_config[key] = value

        return decrypted_config

    def save_encrypted_config(self, config: Dict[str, Any], filename: str = ".env.encrypted"):
        """암호화된 설정 저장"""
        try:
            encrypted_config = self.encrypt_sensitive_config(config)
            with open(filename, 'w') as f:
                json.dump(encrypted_config, f, indent=2)
            self.logger.info(f"암호화된 설정 저장됨: {filename}")
        except Exception as e:
            self.logger.error(f"설정 저장 실패: {e}")

    def load_encrypted_config(self, filename: str = ".env.encrypted") -> Dict[str, Any]:
        """암호화된 설정 로드"""
        try:
            with open(filename, 'r') as f:
                encrypted_config = json.load(f)
            return self.decrypt_sensitive_config(encrypted_config)
        except Exception as e:
            self.logger.error(f"설정 로드 실패: {e}")
            return {}

    def cleanup_expired_tokens(self):
        """만료된 토큰 정리"""
        current_time = datetime.now()
        expired_tokens = []

        for token, token_data in self.access_tokens.items():
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if current_time > expires_at:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self.access_tokens[token]

        if expired_tokens:
            self.logger.info(f"만료된 토큰 {len(expired_tokens)}개 정리됨")

    def get_security_report(self) -> Dict[str, Any]:
        """보안 보고서 생성"""
        return {
            'timestamp': datetime.now().isoformat(),
            'api_keys_count': len(self.api_keys),
            'active_tokens_count': len(self.access_tokens),
            'encryption_key_exists': bool(self.encryption_key),
            'security_status': 'secure' if self.encryption_key else 'insecure'
        }

class SecureConfigManager:
    """보안 설정 관리자"""

    def __init__(self):
        self.security_manager = SecurityManager()
        self.logger = logger

    def load_secure_config(self) -> Dict[str, Any]:
        """보안 설정 로드"""
        # 환경 변수에서 기본 설정 로드
        config = {
            'KIS_APP_KEY': os.getenv('KIS_APP_KEY'),
            'KIS_APP_SECRET': os.getenv('KIS_APP_SECRET'),
            'KIS_ACCESS_TOKEN': os.getenv('KIS_ACCESS_TOKEN'),
            'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
            'DATABASE_URL': os.getenv('DATABASE_URL'),
            'REDIS_URL': os.getenv('REDIS_URL')
        }

        # 암호화된 설정 파일이 있으면 로드
        if os.path.exists('.env.encrypted'):
            encrypted_config = self.security_manager.load_encrypted_config()
            config.update(encrypted_config)

        return config

    def save_secure_config(self, config: Dict[str, Any]):
        """보안 설정 저장"""
        self.security_manager.save_encrypted_config(config)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """설정 검증"""
        required_keys = ['KIS_APP_KEY', 'KIS_APP_SECRET']

        for key in required_keys:
            if not config.get(key):
                self.logger.warning(f"필수 설정 누락: {key}")
                return False

        return True
