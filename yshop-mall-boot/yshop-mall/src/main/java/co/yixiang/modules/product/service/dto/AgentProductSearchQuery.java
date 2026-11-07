package co.yixiang.modules.product.service.dto;

import lombok.Getter;

import java.math.BigDecimal;

/** Validated, immutable parameters for an ordinary-product SKU budget query. */
@Getter
public class AgentProductSearchQuery {
    private final String keyword;
    private final String keywordPattern;
    private final Long categoryId;
    private final BigDecimal minPrice;
    private final BigDecimal maxPrice;
    private final int page;
    private final int limit;
    private final long offset;

    public AgentProductSearchQuery(String keyword, Long categoryId, BigDecimal minPrice, BigDecimal maxPrice,
                                   int page, int limit) {
        String normalized = keyword == null ? "" : keyword.strip();
        if (normalized.codePointCount(0, normalized.length()) > 100) {
            throw new IllegalArgumentException("keyword must contain at most 100 characters");
        }
        if (categoryId != null && categoryId <= 0) {
            throw new IllegalArgumentException("category_id must be a positive integer");
        }
        if ((minPrice != null && minPrice.signum() < 0) || (maxPrice != null && maxPrice.signum() < 0)) {
            throw new IllegalArgumentException("prices must be non-negative decimal amounts");
        }
        if (minPrice != null && maxPrice != null && minPrice.compareTo(maxPrice) > 0) {
            throw new IllegalArgumentException("min_price must not exceed max_price");
        }
        if (page < 1 || limit < 1 || limit > 20) {
            throw new IllegalArgumentException("page must be positive and limit must be between 1 and 20");
        }
        this.keyword = normalized;
        // Use an explicit escape character so SQL does not depend on MySQL backslash modes.
        this.keywordPattern = normalized.isEmpty() ? null : "%" + normalized
                .replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%";
        this.categoryId = categoryId;
        this.minPrice = minPrice;
        this.maxPrice = maxPrice;
        this.page = page;
        this.limit = limit;
        this.offset = ((long) page - 1) * limit;
    }
}
