from typing import Literal


_ERRORS = {
    "action_not_found": (404, "动作不存在或无法访问。"),
    "action_conflict": (409, "动作状态已更新，请勿重复提交。"),
    "action_expired": (409, "动作确认已过期。"),
    "session_not_found": (404, "会话不存在或无法访问，请重新开始会话。"),
    "session_busy": (409, "此会话正在处理另一条消息，请稍后重试。"),
    "session_conflict": (409, "会话状态已更新，请稍后重试。"),
    "session_unavailable": (503, "会话保存暂时不可用，请稍后重试。"),
    "model_not_configured": (503, "生成模型尚未配置，请设置 LLM_BASE_URL、LLM_MODEL 和 LLM_API_KEY。"),
    "model_timeout": (504, "生成模型请求超时，请稍后重试。"),
    "model_unavailable": (502, "生成模型暂时无法访问，请稍后重试。"),
    "model_rejected": (502, "生成模型未能完成请求。"),
    "model_invalid_response": (502, "生成模型返回的数据无效，请稍后重试。"),
    "rag_not_built": (503, "演示规则索引尚未建立，请运行 python -m scripts.build_knowledge 后重试。"),
    "rag_index_incompatible": (503, "演示规则索引与当前模型或资料不兼容，请运行 python -m scripts.build_knowledge 重建。"),
    "rag_index_corrupt": (503, "演示规则索引缺失或损坏，请运行 python -m scripts.build_knowledge 重建。"),
    "rag_model_unavailable": (503, "本地中文模型不可用，请运行 python -m scripts.build_knowledge 准备模型并重建。"),
    "rag_unavailable": (503, "演示规则检索暂时不可用，请运行 python -m scripts.build_knowledge 检查并重建。"),
    "login_required": (401, "请先登录商城，或重新登录后重试。"),
    "invalid_request": (422, "请求参数无效。"),
    "shop_unavailable": (502, "商城暂时无法访问，请稍后重试。"),
    "shop_invalid_response": (502, "商城返回的数据无效，请稍后重试。"),
    "shop_rejected": (502, "商城未能完成请求。"),
    "unsupported_product": (422, "当前仅支持普通商品。"),
    "cart_result_unknown": (502, "加购结果暂时无法确认，请查看购物车，勿重复提交。"),
    "sku_unavailable": (422, "指定规格不属于该商品或已失效。"),
    "insufficient_stock": (422, "当前规格库存不足，已有购物车数量也计入校验。"),
    "cart_state_ambiguous": (422, "购物车存在多条匹配记录，请先检查商城购物车。"),
    "cart_item_invalid": (422, "该规格已有失效购物车记录，请先检查商城购物车。"),
    "cart_verification_failed": (502, "写后购物车未符合预期，请查看购物车，勿重复提交。"),
}


class ShopError(Exception):
    """A public error whose message never includes upstream response data."""

    def __init__(
        self,
        code: str,
        http_status: int | None = None,
        *,
        outcome: Literal["failed", "unknown"] | None = None,
    ) -> None:
        self.code = code if code in _ERRORS else "shop_invalid_response"
        default_status, self.message = _ERRORS[self.code]
        self.http_status = (
            http_status if type(http_status) is int and 400 <= http_status <= 599
            else default_status
        )
        self.outcome = outcome if outcome in {"failed", "unknown"} else None
        super().__init__(self.message)
