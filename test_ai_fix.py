#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트용 파일 - AI 자동수정 기능 검증용
의도적으로 여러 오류를 포함하여 2단계 AI 자동수정이 동작하는지 확인
"""

# 문법 오류: 누락된 콜론
def test_function_without_colon():
    print("이 함수는 콜론이 없어서 문법 오류")

# 들여쓰기 오류
def test_indentation_error():
    print("들여쓰기가 잘못됨")

# 괄호 균형 오류
def test_bracket_error():
    if True:
        print("괄호가 맞지 않음")

# 후행 공백 (1단계에서 수정됨)
def test_trailing_whitespace():
    print("이 줄 끝에 공백이 있음")


def test_too_many_empty_lines():
    print("빈 줄이 너무 많음")


def test_undefined_variable():
    print(undefined_variable)

# 잘못된 import 문
# import invalid_module_name
