# 商城接入范围

仓库保留完整 `yshop-agent`，以及新增的 Java / H5 接入代码。完整商城后台、原有商品与交易页面、数据库、应用密钥和构建产物由已有商城环境提供。

## Java 接入

`yshop-mall-boot` 中的 7 个实现文件和 2 个测试文件保留原包路径，合入对应版本的 YShop 工程：

| 模块 | 内容 |
| --- | --- |
| `yshop-app` | [AgentProductController](../../yshop-mall-boot/yshop-app/src/main/java/co/yixiang/modules/product/rest/AgentProductController.java) 及接口测试 |
| `yshop-mall` | [查询服务](../../yshop-mall-boot/yshop-mall/src/main/java/co/yixiang/modules/product/service/AgentProductSearchService.java)、[Mapper SQL](../../yshop-mall-boot/yshop-mall/src/main/java/co/yixiang/modules/product/service/mapper/AgentProductMapper.java)、4 个 DTO 及查询测试 |

这些文件依赖原商城的 `ApiResult` / `ApiCode`、商品表、鉴权配置和 Maven 模块，以及 Spring Boot、MyBatis、Lombok 等已有依赖。应在自己的完整商城中编译并运行 `AgentProductControllerTest` 与 `AgentProductSearchTest`；仓库中的局部目录没有独立 Maven 构建入口。

Python 适配器使用以下商城接口，均相对 `YSHOP_API_URL`：

| 接口 | 用途 |
| --- | --- |
| `GET userinfo` | 校验身份并取得用户 ID |
| `GET agent/products` | 新增的可购 SKU 预算查询 |
| `GET product/detail/{product_id}` | 实时商品与规格详情 |
| `GET cart/list` | 读取购物车并核对写入结果 |
| `POST cart/add` | 用户确认后的普通加购 |

适配器同时校验 HTTP 状态、商城业务状态和响应字段。接入其他商城时，应调整 [YShopClient](../../yshop-agent/app/tools/yshop.py) 并保持对应的契约测试。

## H5 接入

`yshop-mall-uniapp` 中保留聊天页、3 个卡片组件、API 适配、客户端 / 会话 / SSE 模块，以及 4 个功能测试文件和测试夹具。合入已有 Vue 2 / uni-app 工程时：

1. 按原相对目录复制页面、组件、`api/agent.js` 和 `utils/agent`。
2. 在 H5 的 `pages.json` 中注册以下页面配置。
3. 在现有 `config/index.js` 导出 Agent 服务地址。
4. 沿用商城的 `@/components/uni-icons/uni-icons.vue`、`@/utils/store/cookie` 和 `/pages/user/Login/index` 登录页；如果工程不同，需要替换这些适配点。
5. 安装与本项目一致的 `eventsource-parser@3.1.0`，并使用已有商城的 H5 编译配置。

```json
{
  "path": "pages/agent/Chat/index",
  "style": {
    "navigationBarTitleText": "商城导购",
    "navigationStyle": "custom",
    "disableScroll": true
  }
}
```

```javascript
export const VUE_APP_AGENT_API_URL =
  process.env.VUE_APP_AGENT_API_URL || 'http://127.0.0.1:8018'
```

`api/agent.js` 从商城的 `login_status` 读取当前登录凭据，不把凭据写入 Agent 对话缓存。Agent 的 `CORS_ORIGINS` 必须包含 H5 实际 origin。

核心模块测试不需要启动商城。安装上述 SSE 依赖后，在包含这些文件的 `yshop-mall-uniapp` 目录执行：

```powershell
node --test tests/agent/client.test.mjs tests/agent/conversation.test.mjs tests/agent/durable-lock.test.mjs tests/agent/sse.test.mjs
```

测试覆盖请求协议、会话恢复、确认锁和流式事件处理。原商城的完整页面、浏览器录制脚本和编译验收工具未随本展示仓库发布。
