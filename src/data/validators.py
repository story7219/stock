"""
Data validators for ensuring data quality and consistency.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re

from .models import StockData, MarketData, TechnicalIndicators, DataStatus

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation level enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


class ValidationResult(Enum):
    """Validation result enumeration."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    description: str
    level: ValidationLevel
    validator_func: str
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ValidationIssue:
    """Validation issue details."""
    rule_name: str
    level: ValidationLevel
    result: ValidationResult
    message: str
    details: Dict[str, Any] = None
    affected_fields: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.affected_fields is None:
            self.affected_fields = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    symbol: str
    validation_level: ValidationLevel
    overall_result: ValidationResult
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.overall_result == ValidationResult.PASS
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return any(issue.result == ValidationResult.WARNING for issue in self.issues)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are errors."""
        return any(issue.result == ValidationResult.FAIL for issue in self.issues)


class DataValidator:
    """Comprehensive data validator with configurable rules."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize data validator."""
        self.validation_level = validation_level
        self.rules = self._initialize_rules()
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warnings_count': 0
        }
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize validation rules."""
        rules = []
        
        # Basic validation rules
        rules.extend([
            ValidationRule(
                name="symbol_format",
                description="Validate stock symbol format",
                level=ValidationLevel.BASIC,
                validator_func="validate_symbol_format"
            ),
            ValidationRule(
                name="data_presence",
                description="Check if data is present",
                level=ValidationLevel.BASIC,
                validator_func="validate_data_presence"
            ),
            ValidationRule(
                name="required_columns",
                description="Check for required columns",
                level=ValidationLevel.BASIC,
                validator_func="validate_required_columns",
                parameters={"required_columns": ["Open", "High", "Low", "Close", "Volume"]}
            ),
            ValidationRule(
                name="data_types",
                description="Validate data types",
                level=ValidationLevel.BASIC,
                validator_func="validate_data_types"
            )
        ])
        
        # Standard validation rules
        if self.validation_level.value in ["standard", "strict"]:
            rules.extend([
                ValidationRule(
                    name="price_consistency",
                    description="Check price consistency (High >= Low, etc.)",
                    level=ValidationLevel.STANDARD,
                    validator_func="validate_price_consistency"
                ),
                ValidationRule(
                    name="volume_validity",
                    description="Check volume validity",
                    level=ValidationLevel.STANDARD,
                    validator_func="validate_volume_validity"
                ),
                ValidationRule(
                    name="date_continuity",
                    description="Check date continuity and gaps",
                    level=ValidationLevel.STANDARD,
                    validator_func="validate_date_continuity",
                    parameters={"max_gap_days": 10}
                ),
                ValidationRule(
                    name="outlier_detection",
                    description="Detect statistical outliers",
                    level=ValidationLevel.STANDARD,
                    validator_func="validate_outliers",
                    parameters={"threshold": 3.0}
                ),
                ValidationRule(
                    name="missing_data",
                    description="Check for excessive missing data",
                    level=ValidationLevel.STANDARD,
                    validator_func="validate_missing_data",
                    parameters={"max_missing_ratio": 0.1}
                )
            ])
        
        # Strict validation rules
        if self.validation_level == ValidationLevel.STRICT:
            rules.extend([
                ValidationRule(
                    name="price_range_validation",
                    description="Validate price ranges against historical norms",
                    level=ValidationLevel.STRICT,
                    validator_func="validate_price_ranges"
                ),
                ValidationRule(
                    name="volume_patterns",
                    description="Check for unusual volume patterns",
                    level=ValidationLevel.STRICT,
                    validator_func="validate_volume_patterns"
                ),
                ValidationRule(
                    name="correlation_check",
                    description="Check correlation with market indices",
                    level=ValidationLevel.STRICT,
                    validator_func="validate_correlation_check"
                )
            ])
        
        return rules
    
    async def validate_stock_data(self, stock_data: StockData) -> ValidationReport:
        """Validate individual stock data."""
        issues = []
        
        # Run validation rules
        for rule in self.rules:
            try:
                validator_method = getattr(self, rule.validator_func)
                issue = await validator_method(stock_data, rule)
                if issue:
                    issues.append(issue)
            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                issues.append(ValidationIssue(
                    rule_name=rule.name,
                    level=rule.level,
                    result=ValidationResult.FAIL,
                    message=f"Validation rule execution failed: {str(e)}"
                ))
        
        # Determine overall result
        overall_result = self._determine_overall_result(issues)
        
        # Generate statistics
        statistics = self._generate_statistics(stock_data, issues)
        
        # Update validation stats
        self.validation_stats['total_validations'] += 1
        if overall_result == ValidationResult.PASS:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        self.validation_stats['warnings_count'] += sum(
            1 for issue in issues if issue.result == ValidationResult.WARNING
        )
        
        return ValidationReport(
            symbol=stock_data.symbol,
            validation_level=self.validation_level,
            overall_result=overall_result,
            issues=issues,
            statistics=statistics
        )
    
    async def validate_market_data(self, market_data: MarketData) -> Dict[str, ValidationReport]:
        """Validate entire market data."""
        validation_reports = {}
        
        # Validate each stock
        for stock in market_data.stocks:
            report = await self.validate_stock_data(stock)
            validation_reports[stock.symbol] = report
        
        return validation_reports
    
    async def validate_symbol_format(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Validate stock symbol format."""
        symbol = stock_data.symbol
        
        if not symbol:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.FAIL,
                message="Stock symbol is empty or None",
                affected_fields=["symbol"]
            )
        
        # Basic symbol format validation
        if not re.match(r'^[A-Z0-9._-]+$', symbol):
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.WARNING,
                message=f"Symbol '{symbol}' contains unusual characters",
                affected_fields=["symbol"]
            )
        
        # Length validation
        if len(symbol) > 10:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.WARNING,
                message=f"Symbol '{symbol}' is unusually long ({len(symbol)} characters)",
                affected_fields=["symbol"]
            )
        
        return None
    
    async def validate_data_presence(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check if data is present."""
        if stock_data.data is None:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.FAIL,
                message="No data available",
                affected_fields=["data"]
            )
        
        if stock_data.data.empty:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.FAIL,
                message="Data is empty",
                affected_fields=["data"]
            )
        
        return None
    
    async def validate_required_columns(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check for required columns."""
        if stock_data.data is None or stock_data.data.empty:
            return None  # Will be caught by data_presence validation
        
        required_columns = rule.parameters.get("required_columns", [])
        missing_columns = [col for col in required_columns if col not in stock_data.data.columns]
        
        if missing_columns:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.FAIL,
                message=f"Missing required columns: {missing_columns}",
                affected_fields=missing_columns,
                details={"missing_columns": missing_columns}
            )
        
        return None
    
    async def validate_data_types(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Validate data types."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        issues = []
        
        # Check numeric columns
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"{col} is not numeric")
        
        if issues:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.FAIL,
                message=f"Data type issues: {'; '.join(issues)}",
                affected_fields=[col for col in numeric_columns if col in df.columns],
                details={"type_issues": issues}
            )
        
        return None
    
    async def validate_price_consistency(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check price consistency."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        required_cols = ["Open", "High", "Low", "Close"]
        
        if not all(col in df.columns for col in required_cols):
            return None  # Will be caught by required_columns validation
        
        # Check High >= Low
        high_low_violations = (df["High"] < df["Low"]).sum()
        
        # Check High >= Open, Close
        high_open_violations = (df["High"] < df["Open"]).sum()
        high_close_violations = (df["High"] < df["Close"]).sum()
        
        # Check Low <= Open, Close
        low_open_violations = (df["Low"] > df["Open"]).sum()
        low_close_violations = (df["Low"] > df["Close"]).sum()
        
        total_violations = (high_low_violations + high_open_violations + 
                          high_close_violations + low_open_violations + low_close_violations)
        
        if total_violations > 0:
            violation_ratio = total_violations / len(df)
            
            result = ValidationResult.FAIL if violation_ratio > 0.05 else ValidationResult.WARNING
            
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=result,
                message=f"Price consistency violations: {total_violations} out of {len(df)} records ({violation_ratio:.2%})",
                affected_fields=required_cols,
                details={
                    "high_low_violations": int(high_low_violations),
                    "high_open_violations": int(high_open_violations),
                    "high_close_violations": int(high_close_violations),
                    "low_open_violations": int(low_open_violations),
                    "low_close_violations": int(low_close_violations),
                    "violation_ratio": violation_ratio
                }
            )
        
        return None
    
    async def validate_volume_validity(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check volume validity."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        
        if "Volume" not in df.columns:
            return None
        
        # Check for negative volumes
        negative_volumes = (df["Volume"] < 0).sum()
        
        # Check for zero volumes (might be suspicious)
        zero_volumes = (df["Volume"] == 0).sum()
        
        # Check for extremely high volumes (outliers)
        if len(df) > 10:
            volume_median = df["Volume"].median()
            volume_q99 = df["Volume"].quantile(0.99)
            extreme_volumes = (df["Volume"] > volume_q99 * 10).sum()
        else:
            extreme_volumes = 0
        
        issues = []
        if negative_volumes > 0:
            issues.append(f"{negative_volumes} negative volumes")
        
        if zero_volumes > len(df) * 0.1:  # More than 10% zero volumes
            issues.append(f"{zero_volumes} zero volumes ({zero_volumes/len(df):.1%})")
        
        if extreme_volumes > 0:
            issues.append(f"{extreme_volumes} extremely high volumes")
        
        if issues:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.WARNING,
                message=f"Volume issues: {'; '.join(issues)}",
                affected_fields=["Volume"],
                details={
                    "negative_volumes": int(negative_volumes),
                    "zero_volumes": int(zero_volumes),
                    "extreme_volumes": int(extreme_volumes)
                }
            )
        
        return None
    
    async def validate_date_continuity(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check date continuity and gaps."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        max_gap_days = rule.parameters.get("max_gap_days", 10)
        
        if len(df) < 2:
            return None
        
        # Calculate date gaps
        date_index = pd.to_datetime(df.index)
        date_diffs = date_index.to_series().diff().dt.days
        
        # Find large gaps
        large_gaps = date_diffs[date_diffs > max_gap_days]
        
        if len(large_gaps) > 0:
            max_gap = large_gaps.max()
            gap_count = len(large_gaps)
            
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.WARNING,
                message=f"Found {gap_count} date gaps larger than {max_gap_days} days (max gap: {max_gap} days)",
                affected_fields=["index"],
                details={
                    "gap_count": int(gap_count),
                    "max_gap_days": int(max_gap),
                    "threshold": max_gap_days
                }
            )
        
        return None
    
    async def validate_outliers(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Detect statistical outliers."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        threshold = rule.parameters.get("threshold", 3.0)
        
        # Calculate returns for outlier detection
        if "Close" in df.columns and len(df) > 10:
            returns = df["Close"].pct_change().dropna()
            
            if len(returns) > 0:
                # Use z-score method
                z_scores = np.abs((returns - returns.mean()) / returns.std())
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_ratio = outlier_count / len(returns)
                    max_z_score = z_scores.max()
                    
                    result = ValidationResult.FAIL if outlier_ratio > 0.05 else ValidationResult.WARNING
                    
                    return ValidationIssue(
                        rule_name=rule.name,
                        level=rule.level,
                        result=result,
                        message=f"Found {outlier_count} outliers ({outlier_ratio:.2%}) with max z-score: {max_z_score:.2f}",
                        affected_fields=["Close"],
                        details={
                            "outlier_count": int(outlier_count),
                            "outlier_ratio": outlier_ratio,
                            "max_z_score": float(max_z_score),
                            "threshold": threshold
                        }
                    )
        
        return None
    
    async def validate_missing_data(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check for excessive missing data."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        max_missing_ratio = rule.parameters.get("max_missing_ratio", 0.1)
        
        # Calculate missing data ratio
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells
        
        if missing_ratio > max_missing_ratio:
            return ValidationIssue(
                rule_name=rule.name,
                level=rule.level,
                result=ValidationResult.WARNING,
                message=f"High missing data ratio: {missing_ratio:.2%} (threshold: {max_missing_ratio:.2%})",
                affected_fields=df.columns.tolist(),
                details={
                    "missing_ratio": missing_ratio,
                    "missing_cells": int(missing_cells),
                    "total_cells": int(total_cells),
                    "threshold": max_missing_ratio
                }
            )
        
        return None
    
    async def validate_price_ranges(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Validate price ranges against historical norms."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        
        if "Close" in df.columns and len(df) > 50:
            prices = df["Close"].dropna()
            
            # Check for unrealistic price movements
            returns = prices.pct_change().dropna()
            
            # Daily returns should typically be within reasonable bounds
            extreme_returns = returns.abs() > 0.5  # 50% daily change
            extreme_count = extreme_returns.sum()
            
            if extreme_count > 0:
                return ValidationIssue(
                    rule_name=rule.name,
                    level=rule.level,
                    result=ValidationResult.WARNING,
                    message=f"Found {extreme_count} extreme price movements (>50% daily change)",
                    affected_fields=["Close"],
                    details={
                        "extreme_count": int(extreme_count),
                        "max_return": float(returns.abs().max()),
                        "threshold": 0.5
                    }
                )
        
        return None
    
    async def validate_volume_patterns(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check for unusual volume patterns."""
        if stock_data.data is None or stock_data.data.empty:
            return None
        
        df = stock_data.data
        
        if "Volume" in df.columns and len(df) > 20:
            volumes = df["Volume"].dropna()
            
            if len(volumes) > 0:
                # Check for volume spikes
                volume_mean = volumes.mean()
                volume_std = volumes.std()
                
                if volume_std > 0:
                    z_scores = (volumes - volume_mean) / volume_std
                    volume_spikes = (z_scores > 5).sum()  # More than 5 standard deviations
                    
                    if volume_spikes > 0:
                        return ValidationIssue(
                            rule_name=rule.name,
                            level=rule.level,
                            result=ValidationResult.WARNING,
                            message=f"Found {volume_spikes} unusual volume spikes",
                            affected_fields=["Volume"],
                            details={
                                "volume_spikes": int(volume_spikes),
                                "max_z_score": float(z_scores.max()),
                                "threshold": 5.0
                            }
                        )
        
        return None
    
    async def validate_correlation_check(self, stock_data: StockData, rule: ValidationRule) -> Optional[ValidationIssue]:
        """Check correlation with market indices (placeholder)."""
        # This would require market index data for comparison
        # For now, return None (no validation)
        return None
    
    def _determine_overall_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Determine overall validation result."""
        if not issues:
            return ValidationResult.PASS
        
        # If any issue is FAIL, overall result is FAIL
        if any(issue.result == ValidationResult.FAIL for issue in issues):
            return ValidationResult.FAIL
        
        # If all issues are warnings, overall result is WARNING
        if all(issue.result == ValidationResult.WARNING for issue in issues):
            return ValidationResult.WARNING
        
        return ValidationResult.PASS
    
    def _generate_statistics(self, stock_data: StockData, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate validation statistics."""
        stats = {
            "total_rules_checked": len(self.rules),
            "issues_found": len(issues),
            "warnings": sum(1 for issue in issues if issue.result == ValidationResult.WARNING),
            "errors": sum(1 for issue in issues if issue.result == ValidationResult.FAIL),
            "data_points": len(stock_data.data) if stock_data.data is not None else 0,
            "data_columns": len(stock_data.data.columns) if stock_data.data is not None else 0
        }
        
        return stats
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get overall validation statistics."""
        return self.validation_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warnings_count': 0
        } 