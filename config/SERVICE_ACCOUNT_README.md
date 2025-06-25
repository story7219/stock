# 🔐 Google Cloud Service Account 인증 파일 가이드

## 📋 개요

`SERVICE_ACCOUNT_JSON` 파일은 Google Cloud Platform (GCP) 서비스 계정의 인증 정보를 포함하는 중요한 보안 파일입니다. 이 파일을 통해 Gemini AI API와 기타 Google Cloud 서비스에 접근할 수 있습니다.

## 🚨 중요한 보안 주의사항

### ⚠️ 절대 금지 사항
- **Git 저장소에 커밋하거나 공개하지 마세요**
- **이메일, 메신저, 공개 포럼에 공유하지 마세요**
- **스크린샷이나 화면 공유 시 노출되지 않도록 주의하세요**
- **클라우드 저장소(Google Drive, Dropbox 등)에 업로드하지 마세요**

### 🔒 보안 설정 체크리스트
- [ ] 파일 권한을 600 (소유자만 읽기/쓰기)으로 설정
- [ ] `.gitignore`에 이 파일이 포함되어 있는지 확인
- [ ] 불필요한 권한이 부여되지 않았는지 확인
- [ ] 정기적인 키 로테이션 계획 수립
- [ ] 프로덕션 환경에서는 환경 변수 사용 권장

## 📋 파일 구성 요소

```json
{
  "type": "service_account",           // 계정 유형
  "project_id": "프로젝트-ID",          // GCP 프로젝트 식별자
  "private_key_id": "키-ID",           // 개인키 식별자
  "private_key": "-----BEGIN...",      // RSA 개인키 (PEM 형식)
  "client_email": "이메일@project.iam", // 서비스 계정 이메일
  "client_id": "클라이언트-ID",         // OAuth2 클라이언트 ID
  "auth_uri": "인증-URI",              // OAuth2 인증 엔드포인트
  "token_uri": "토큰-URI",             // OAuth2 토큰 엔드포인트
  "auth_provider_x509_cert_url": "...", // 인증 제공자 인증서 URL
  "client_x509_cert_url": "...",       // 클라이언트 인증서 URL
  "universe_domain": "googleapis.com"  // API 도메인
}
```

## 🔧 사용 방법

### 1. 환경 변수 설정 (권장)
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/SERVICE_ACCOUNT_JSON"
```

### 2. 코드에서 직접 사용
```python
import os
from google.oauth2 import service_account

# 인증 정보 로드
credentials = service_account.Credentials.from_service_account_file(
    'config/SERVICE_ACCOUNT_JSON'
)

# Gemini AI 클라이언트 초기화
import google.generativeai as genai
genai.configure(credentials=credentials)
```

### 3. 자동 인증 (ADC)
```python
from google.auth import default

# 환경 변수나 기본 위치에서 자동으로 인증 정보 찾기
credentials, project = default()
```

## 🛠️ 파일 권한 설정

### Linux/macOS
```bash
chmod 600 config/SERVICE_ACCOUNT_JSON
```

### Windows (PowerShell)
```powershell
icacls config\SERVICE_ACCOUNT_JSON /inheritance:r /grant:r "%USERNAME%:F"
```

## 🔄 키 로테이션 (정기 교체)

### 1. GCP 콘솔에서 새 키 생성
1. [GCP 콘솔](https://console.cloud.google.com/) 접속
2. IAM 및 관리자 > 서비스 계정 선택
3. 해당 서비스 계정 클릭
4. "키" 탭에서 "키 추가" > "새 키 만들기" 선택
5. JSON 형식으로 다운로드

### 2. 기존 키 삭제
- 새 키가 정상 작동하는지 확인 후 기존 키 삭제
- 보안상 불필요한 키는 즉시 삭제 권장

## 🚨 보안 사고 발생 시 대응

### 키가 노출되었다고 의심되는 경우
1. **즉시 해당 키를 GCP 콘솔에서 삭제**
2. **새로운 키 생성 및 교체**
3. **접근 로그 확인 및 이상 활동 모니터링**
4. **필요시 프로젝트 전체 보안 점검**

## 💡 모범 사례

### 개발 환경
- 개발용 서비스 계정과 프로덕션용 서비스 계정 분리
- 최소 권한 원칙 적용 (필요한 권한만 부여)
- 로컬 개발 시에만 파일 사용, CI/CD에서는 환경 변수 사용

### 프로덕션 환경
- 환경 변수나 보안 관리 서비스 사용 (AWS Secrets Manager, Azure Key Vault 등)
- 컨테이너 환경에서는 Secret 볼륨 마운트
- 정기적인 키 로테이션 자동화

## 🔍 문제 해결

### 인증 오류 발생 시
1. 파일 경로가 정확한지 확인
2. 파일 권한이 적절한지 확인
3. JSON 형식이 유효한지 확인
4. 서비스 계정에 필요한 권한이 있는지 확인

### 권한 부족 오류 시
1. GCP 콘솔에서 서비스 계정 권한 확인
2. 필요한 API가 활성화되어 있는지 확인
3. 프로젝트 수준에서 적절한 역할이 할당되어 있는지 확인

---

**⚠️ 이 파일은 매우 중요한 보안 자산입니다. 항상 신중하게 다루고 보안 모범 사례를 준수하세요.** 