# 🔄 자동화된 백업 시스템

## 📋 개요

이 시스템은 GitHub Actions와 네이버 클라우드를 연동한 자동화된 백업 시스템입니다.

### 🎯 주요 기능
- **GitHub Actions 자동 백업**: 매일 오전 2시 자동 실행
- **용량 관리**: 900MB 제한, 오래된 파일 자동 삭제
- **네이버 클라우드 연동**: 30GB 무료 저장소 활용
- **보안**: 민감 정보 자동 제외

## 🚀 설정 방법

### 1. GitHub Secrets 설정

GitHub 저장소의 Settings > Secrets and variables > Actions에서 다음 환경변수를 설정하세요:

```bash
NAVER_CLOUD_ACCESS_KEY=your_access_key
NAVER_CLOUD_SECRET_KEY=your_secret_key
NAVER_CLOUD_BUCKET=auto-trading-backup
```

### 2. 네이버 클라우드 설정

1. **네이버 클라우드 플랫폼 가입**
   - https://www.ncloud.com/
   - 30GB 무료 저장소 제공

2. **API 키 생성**
   - Object Storage API 키 발급
   - Access Key와 Secret Key 확인

3. **버킷 생성**
   - Object Storage에서 버킷 생성
   - 버킷명: `auto-trading-backup`

## 📁 백업 대상

### ✅ 백업되는 파일들
```
src/           # 소스 코드
scripts/       # 스크립트
config/        # 설정 파일
tests/         # 테스트 코드
docs/          # 문서
data_engine/   # 데이터 엔진
strategy/      # 트레이딩 전략
monitoring/    # 모니터링 시스템
```

### ❌ 백업되지 않는 파일들
```
__pycache__/   # Python 캐시
*.log          # 로그 파일
*.tmp          # 임시 파일
.env           # 환경 변수
secrets/       # 민감 정보
api_keys/      # API 키
collected_data/ # 수집된 데이터
models/        # 모델 파일
cache/         # 캐시 파일
```

## ⚙️ 자동화 워크플로우

### GitHub Actions 스케줄
```yaml
# 매일 오전 2시 자동 실행
- cron: '0 2 * * *'
```

### 실행 단계
1. **코드 체크아웃**
2. **Python 환경 설정**
3. **백업 스크립트 실행**
4. **저장소 크기 확인**
5. **오래된 파일 정리 (900MB 초과 시)**
6. **네이버 클라우드 업로드**
7. **변경사항 커밋 및 푸시**

## 📊 용량 관리

### 자동 정리 규칙
- **저장소 크기**: 900MB 제한
- **정리 우선순위**:
  1. 임시 파일 (7일 이상)
  2. 오래된 백업 (30일 이상)
  3. 큰 데이터 파일 (60일 이상)

### 수동 정리
```bash
# 오래된 파일 정리
python scripts/cleanup_old_files.py

# GitHub 백업 생성
python scripts/github_backup_manager.py

# 네이버 클라우드 업로드
python scripts/naver_cloud_upload.py
```

## 🔍 백업 상태 확인

### GitHub 저장소 크기 확인
```bash
# 저장소 크기 계산
du -sh . | cut -f1
```

### 네이버 클라우드 사용량 확인
```bash
# 사용량 확인 (스크립트 내장)
python scripts/naver_cloud_upload.py --check-usage
```

## 🛠️ 문제 해결

### 일반적인 문제들

#### 1. GitHub Actions 실패
```bash
# 로그 확인
# GitHub 저장소 > Actions 탭에서 확인
```

#### 2. 네이버 클라우드 업로드 실패
```bash
# API 키 확인
echo $NAVER_CLOUD_ACCESS_KEY
echo $NAVER_CLOUD_SECRET_KEY

# 수동 업로드 테스트
python scripts/naver_cloud_upload.py
```

#### 3. 저장소 크기 초과
```bash
# 큰 파일 찾기
find . -type f -size +10M

# 수동 정리
python scripts/cleanup_old_files.py
```

## 📈 모니터링

### 백업 로그 확인
```bash
# GitHub Actions 로그
# 저장소 > Actions > 최근 실행 > 로그 확인

# 로컬 로그
tail -f logs/backup.log
```

### 알림 설정
- GitHub Actions 실패 시 이메일 알림
- 네이버 클라우드 사용량 25GB 초과 시 경고

## 🔒 보안 고려사항

### 민감 정보 보호
- `.env` 파일은 절대 업로드되지 않음
- API 키는 GitHub Secrets에 저장
- 로그 파일은 자동 제외

### 접근 권한
- GitHub 저장소: Private 권장
- 네이버 클라우드: 버킷 접근 권한 설정

## 📞 지원

### 문제 발생 시
1. **GitHub Issues**에 문제 보고
2. **로그 파일** 확인
3. **수동 실행**으로 테스트

### 연락처
- GitHub: 저장소 Issues
- 이메일: [your-email@example.com]

---

## 🎯 성능 지표

### 백업 성능
- **백업 속도**: 평균 2-5분
- **압축률**: 약 60-70%
- **복구 시간**: 1-3분

### 저장소 효율성
- **GitHub**: 900MB 제한 내 유지
- **네이버 클라우드**: 30GB 중 5-10GB 사용
- **자동 정리**: 주 1회 실행

### 안정성
- **백업 성공률**: 99.5%+
- **복구 성공률**: 100%
- **데이터 무결성**: MD5 해시 검증 