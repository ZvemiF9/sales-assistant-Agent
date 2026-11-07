package co.yixiang.modules.product.rest;

import co.yixiang.api.ApiCode;
import co.yixiang.api.ApiResult;
import co.yixiang.modules.product.service.AgentProductSearchService;
import co.yixiang.modules.product.service.dto.AgentProductSearchQuery;
import co.yixiang.modules.product.service.dto.AgentProductSearchResult;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

import java.math.BigDecimal;

@RestController
@RequiredArgsConstructor
@Tag(name = "商城:Agent商品查询")
public class AgentProductController {
    private final AgentProductSearchService searchService;

    @GetMapping("/agent/products")
    @Operation(summary = "按可购规格价格查询普通商品")
    public ApiResult<AgentProductSearchResult> search(
            @RequestParam(value = "keyword", defaultValue = "") String keyword,
            @RequestParam(value = "category_id", required = false) Long categoryId,
            @RequestParam(value = "min_price", required = false) BigDecimal minPrice,
            @RequestParam(value = "max_price", required = false) BigDecimal maxPrice,
            @RequestParam(value = "page", defaultValue = "1") int page,
            @RequestParam(value = "limit", defaultValue = "10") int limit) {
        return ApiResult.ok(searchService.search(
                new AgentProductSearchQuery(keyword, categoryId, minPrice, maxPrice, page, limit)));
    }

    @ExceptionHandler({IllegalArgumentException.class, MethodArgumentTypeMismatchException.class})
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ApiResult<String> invalidParameters(Exception exception) {
        return ApiResult.fail(ApiCode.PARAMETER_EXCEPTION.getCode(), "Invalid product search parameters");
    }
}
