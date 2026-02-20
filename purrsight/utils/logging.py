"""
日志配置模块：提供统一的日志配置和MLflowLogger兼容性处理

包含：
- logging_config: 日志配置字典
- logger: 根logger实例
- MLflowLogger: MLflow logger类（兼容性处理）
"""

import logging
import logging.config
import sys
from pathlib import Path
from purrsight.config import LOGS_DIR

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {"format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

# Logger
logging.config.dictConfig(logging_config)
logger = logging.getLogger()



# MLflowLogger 兼容性处理
try:
    from pytorch_lightning.loggers import MLflowLogger as _MLflowLogger
    MLflowLogger = _MLflowLogger  # 使用标准版本
except ImportError:
    try:
        from pytorch_lightning.loggers.mlflow import MLflowLogger as _MLflowLogger
        MLflowLogger = _MLflowLogger  # 使用子模块版本
    except ImportError:
        # 如果MLflowLogger不存在，创建一个自定义的logger包装器
        import mlflow
        from pytorch_lightning.loggers import Logger
        
        class MLflowLogger(Logger):
            """
            简单的MLflow logger包装器
            
            当PyTorch Lightning的MLflowLogger不可用时使用此实现。
            实现了PyTorch Lightning Logger接口，与MLflow集成。
            """
            def __init__(self, experiment_name, tracking_uri=None, run_name=None):
                super().__init__()
                self.experiment_name = experiment_name
                self.tracking_uri = tracking_uri
                self.run_name = run_name
                
                # 设置tracking URI和experiment
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                
                # 检查是否已经有active run，如果有就使用它，否则启动新的
                try:
                    active_run = mlflow.active_run()
                    if active_run is not None:
                        # 使用已有的run
                        self.run = active_run
                        self._own_run = False
                    else:
                        # 启动新的run
                        if run_name:
                            self.run = mlflow.start_run(run_name=run_name)
                        else:
                            self.run = mlflow.start_run()
                        self._own_run = True
                except Exception:
                    # 如果检查失败，尝试启动新的run
                    if run_name:
                        self.run = mlflow.start_run(run_name=run_name)
                    else:
                        self.run = mlflow.start_run()
                    self._own_run = True
            
            @property
            def experiment(self):
                """返回MLflow实验对象"""
                return mlflow
            
            @property
            def version(self):
                """返回运行ID"""
                return self.run.info.run_id
            
            @property
            def name(self):
                """返回实验名称"""
                return self.experiment_name
            
            def log_hyperparams(self, params):
                """记录超参数"""
                mlflow.log_params(params)
            
            def log_metrics(self, metrics, step=None):
                """记录多个指标"""
                if step is not None:
                    metrics = {k: v for k, v in metrics.items()}
                mlflow.log_metrics(metrics, step=step)
            
            def log_metric(self, key, value, step=None):
                """记录单个指标"""
                mlflow.log_metric(key, value, step=step)
            
            def finalize(self, status="success"):
                """结束MLflow运行"""
                # 只有当我们拥有这个run时才结束它
                if hasattr(self, '_own_run') and self._own_run:
                    mlflow.end_run()
