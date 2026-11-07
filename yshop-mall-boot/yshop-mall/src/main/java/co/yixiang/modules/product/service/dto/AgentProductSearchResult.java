package co.yixiang.modules.product.service.dto;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.List;

@Getter
@RequiredArgsConstructor
public class AgentProductSearchResult {
    private final List<AgentProductDto> products;
    private final int page;
    private final int limit;
    private final long total;
}
