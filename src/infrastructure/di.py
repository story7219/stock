from __future__ import annotations

from application.cli import CLIService
from application.dashboard import DashboardService
from application.services import TradingSystemService
from abc import ABC
from abc import abstractmethod
from core.logger import get_logger
from functools import wraps
from typing import Any
import Callable, Dict, Optional, Type, TypeVar, Union, cast
import asyncio
import inspect

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일명: di.py
모듈: 의존성 주입 컨테이너
목적: 서비스 등록, 해결, 생명주기 관리

Author: Trading Strategy System
Created: 2025-01-27
Modified: 2025-01-27
Version: 2.0.0

Dependencies:
    - Python 3.11+
    - typing-extensions==4.8.0
    - asyncio

Architecture:
    - Dependency Injection
    - Service Locator Pattern
    - Lifecycle Management
    - Singleton Pattern

License: MIT
"""




T = TypeVar('T')
logger = get_logger(__name__)


class ServiceLifetime(ABC):
    """서비스 생명주기 추상 클래스"""

    @abstractmethod
    async def get_instance(self, factory: Callable[[], Any]) -> Any:
        """인스턴스 반환"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """정리 작업"""
        pass


class SingletonLifetime(ServiceLifetime):
    """싱글톤 생명주기"""

    def __init__(self) -> None:
        self._instance: Optional[Any] = None
        self._factory: Optional[Callable[[], Any]] = None

    async def get_instance(self, factory: Callable[[], Any]) -> Any:
        """싱글톤 인스턴스 반환"""
        if self._instance is None:
            self._factory = factory
            self._instance = factory()
            logger.debug(f"Created singleton instance: {type(self._instance).__name__}")
        return self._instance

    async def cleanup(self) -> None:
        """싱글톤 정리"""
        if self._instance is not None:
            if hasattr(self._instance, 'cleanup'):
                if asyncio.iscoroutinefunction(self._instance.cleanup):
                    await self._instance.cleanup()
                else:
                    self._instance.cleanup()
            if hasattr(self._instance, 'dispose'):
                if asyncio.iscoroutinefunction(self._instance.dispose):
                    await self._instance.dispose()
                else:
                    self._instance.dispose()
            self._instance = None
            logger.debug("Singleton instance cleaned up")


class TransientLifetime(ServiceLifetime):
    """일시적 생명주기 (매번 새 인스턴스)"""

    async def get_instance(self, factory: Callable[[], Any]) -> Any:
        """새 인스턴스 반환"""
        instance = factory()
        logger.debug(f"Created transient instance: {type(instance).__name__}")
        return instance

    async def cleanup(self) -> None:
        """일시적 서비스는 정리할 것이 없음"""
        pass


class ScopedLifetime(ServiceLifetime):
    """스코프 생명주기 (스코프 내에서 싱글톤)"""

    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    async def get_instance(self, factory: Callable[[], Any]) -> Any:
        scope_id = str(id(asyncio.current_task()))
        if scope_id not in self._instances:
            self._factories[scope_id] = factory
            self._instances[scope_id] = factory()
            logger.debug(f"Created scoped instance: {type(self._instances[scope_id]).__name__}")
        return self._instances[scope_id]

    async def cleanup(self) -> None:
        for scope_id, instance in self._instances.items():
            if hasattr(instance, 'cleanup'):
                if asyncio.iscoroutinefunction(instance.cleanup):
                    await instance.cleanup()
                else:
                    instance.cleanup()
            if hasattr(instance, 'dispose'):
                if asyncio.iscoroutinefunction(instance.dispose):
                    await instance.dispose()
                else:
                    instance.dispose()
        self._instances.clear()
        self._factories.clear()
        logger.debug("Scoped instances cleaned up")


class ServiceDescriptor:
    def __init__(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None, factory: Optional[Callable[[], T]] = None, lifetime: Optional[ServiceLifetime] = None) -> None:
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.lifetime = lifetime or SingletonLifetime()

    def create_factory(self) -> Callable[[], Any]:
        if self.factory:
            return self.factory
        def factory() -> Any:
            return self.implementation_type()
        return factory


class DependencyContainer:
    """의존성 주입 컨테이너"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._initialized = False
        logger.info("DependencyContainer 초기화")

    async def initialize(self) -> None:
        """컨테이너 초기화"""
        if self._initialized:
            return

        logger.info("의존성 주입 컨테이너 초기화 시작")

        # 기본 서비스들 등록
        self.register_singleton(CLIService)
        self.register_singleton(DashboardService)
        self.register_singleton(TradingSystemService)

        # 서비스 인스턴스 생성
        for service_type in self._services:
            await self.resolve(service_type)

        self._initialized = True
        logger.info("의존성 주입 컨테이너 초기화 완료")

    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
        """싱글톤 서비스 등록"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=SingletonLifetime()
        )
        self._services[service_type] = descriptor
        logger.debug(f"싱글톤 서비스 등록: {service_type.__name__}")

    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
        """일시적 서비스 등록"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=TransientLifetime()
        )
        self._services[service_type] = descriptor
        logger.debug(f"일시적 서비스 등록: {service_type.__name__}")

    def register_scoped(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
        """스코프 서비스 등록"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ScopedLifetime()
        )
        self._services[service_type] = descriptor
        logger.debug(f"스코프 서비스 등록: {service_type.__name__}")

    async def resolve(self, service_type: Type[T]) -> T:
        """서비스 해결"""
        if service_type not in self._services:
            raise KeyError(f"등록되지 않은 서비스: {service_type.__name__}")

        descriptor = self._services[service_type]
        instance = await descriptor.lifetime.get_instance(descriptor.create_factory())

        logger.debug(f"서비스 해결: {service_type.__name__}")
        return cast(T, instance)

    async def cleanup(self) -> None:
        """컨테이너 정리"""
        logger.info("의존성 주입 컨테이너 정리 시작")

        for descriptor in self._services.values():
            await descriptor.lifetime.cleanup()

        self._services.clear()
        self._instances.clear()
        self._initialized = False

        logger.info("의존성 주입 컨테이너 정리 완료")

    def get(self, service_type: Type[T]) -> T:
        """서비스 가져오기 (동기, 예외 발생)"""
        if service_type not in self._services:
            raise KeyError(f"등록되지 않은 서비스: {service_type.__name__}")

        descriptor = self._services[service_type]

        # 싱글톤의 경우 이미 생성된 인스턴스가 있는지 확인
        if isinstance(descriptor.lifetime, SingletonLifetime):
            if descriptor.lifetime._instance is not None:
                return cast(T, descriptor.lifetime._instance)

        # 새 인스턴스 생성 (동기적으로)
        instance = descriptor.create_factory()()
        logger.debug(f"서비스 해결 (동기): {service_type.__name__}")

        # 싱글톤의 경우 인스턴스 저장
        if isinstance(descriptor.lifetime, SingletonLifetime):
            descriptor.lifetime._instance = instance

        return cast(T, instance)
