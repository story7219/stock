from __future__ import annotations
from abc import ABC, abstractmethod
from core.logger import get_logger
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast
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
    """서비스 설명자"""

    def __init__(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        lifetime: Optional[ServiceLifetime] = None
    ) -> None:
        self.service_type = service_type
        self.implementation_type = implementation_type or service_type
        self.factory = factory
        self.lifetime = lifetime or SingletonLifetime()

    def create_factory(self) -> Callable[[], Any]:
        """팩토리 함수 생성"""
        if self.factory:
            return self.factory

        def factory() -> Any:
            return self.implementation_type()
        return factory


class DependencyContainer:
    """의존성 주입 컨테이너"""

    def __init__(self) -> None:
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """컨테이너 초기화"""
        if self._initialized:
            return

        logger.info("Initializing dependency container")
        
        # 기본 서비스들 등록
        self._register_default_services()
        
        self._initialized = True
        logger.info("Dependency container initialized")

    def _register_default_services(self) -> None:
        """기본 서비스들 등록"""
        try:
            from application.services import TradingSystemService
            from application.cli import CLIService
            from application.dashboard import DashboardService
            
            # TradingSystemService 등록
            self.register_singleton(TradingSystemService, TradingSystemService)
            
            # CLIService 등록
            self.register_singleton(CLIService, CLIService)
            
            # DashboardService 등록
            self.register_singleton(DashboardService, DashboardService)
            
            logger.info("Default services registered successfully")
            
        except ImportError as e:
            logger.warning(f"Some services could not be imported: {e}")

    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
        """싱글톤 서비스 등록"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=SingletonLifetime()
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered singleton service: {service_type.__name__}")

    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
        """일시적 서비스 등록"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=TransientLifetime()
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered transient service: {service_type.__name__}")

    def register_scoped(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> None:
        """스코프 서비스 등록"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ScopedLifetime()
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered scoped service: {service_type.__name__}")

    def register_factory(self, service_type: Type[T], factory: Callable[[], T], lifetime: Optional[ServiceLifetime] = None) -> None:
        """팩토리 서비스 등록"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            lifetime=lifetime or SingletonLifetime()
        )
        self._services[service_type] = descriptor
        logger.debug(f"Registered factory service: {service_type.__name__}")

    def get(self, service_type: Type[T]) -> T:
        """서비스 인스턴스 반환"""
        if service_type not in self._services:
            raise ValueError(f"Service not registered: {service_type.__name__}")

        descriptor = self._services[service_type]
        
        # 싱글톤인 경우 캐시된 인스턴스 반환
        if isinstance(descriptor.lifetime, SingletonLifetime):
            if service_type not in self._instances:
                factory = descriptor.create_factory()
                self._instances[service_type] = factory()
            return cast(T, self._instances[service_type])
        
        # 다른 생명주기는 팩토리에서 새로 생성
        factory = descriptor.create_factory()
        return cast(T, factory())

    async def get_async(self, service_type: Type[T]) -> T:
        """비동기 서비스 인스턴스 반환"""
        if service_type not in self._services:
            raise ValueError(f"Service not registered: {service_type.__name__}")

        descriptor = self._services[service_type]
        factory = descriptor.create_factory()
        instance = await descriptor.lifetime.get_instance(factory)
        return cast(T, instance)

    def resolve(self, target_type: Type[T]) -> T:
        """의존성 해결 (생성자 주입)"""
        # 간단한 구현 - 실제로는 더 복잡한 의존성 해결이 필요
        return self.get(target_type)

    async def cleanup(self) -> None:
        """컨테이너 정리"""
        logger.info("Cleaning up dependency container")
        
        # 등록된 서비스들 정리
        for service_type, descriptor in self._services.items():
            try:
                await descriptor.lifetime.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up service {service_type.__name__}: {e}")
        
        # 캐시된 인스턴스들 정리
        self._instances.clear()
        self._services.clear()
        self._initialized = False
        
        logger.info("Dependency container cleaned up")


# 전역 컨테이너 인스턴스
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """전역 컨테이너 인스턴스 반환"""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def register_services(container: DependencyContainer) -> None:
    """서비스 등록 헬퍼 함수"""
    # 여기에 추가 서비스 등록 로직을 추가할 수 있습니다
    pass
