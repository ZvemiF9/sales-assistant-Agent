package co.yixiang.modules.product.service;

import co.yixiang.modules.product.service.dto.AgentProductDto;
import co.yixiang.modules.product.service.dto.AgentProductSearchQuery;
import co.yixiang.modules.product.service.dto.AgentProductSearchResult;
import co.yixiang.modules.product.service.dto.AgentProductSkuDto;
import co.yixiang.modules.product.service.mapper.AgentProductMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class AgentProductSearchService {
    private final AgentProductMapper mapper;

    /** A consistent snapshot keeps counts, the product page and its SKUs aligned. */
    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public AgentProductSearchResult search(AgentProductSearchQuery query) {
        long total = mapper.countProducts(query);
        if (total == 0 || query.getOffset() >= total) {
            return new AgentProductSearchResult(List.of(), query.getPage(), query.getLimit(), total);
        }
        List<AgentProductDto> products = mapper.selectProducts(query);
        if (!products.isEmpty()) {
            Map<Long, AgentProductDto> byId = new LinkedHashMap<>();
            products.forEach(product -> byId.put(product.getProductId(), product));
            for (AgentProductSkuDto sku : mapper.selectSkus(query, List.copyOf(byId.keySet()))) {
                byId.get(sku.getProductId()).getSkus().add(sku);
            }
        }
        return new AgentProductSearchResult(products, query.getPage(), query.getLimit(), total);
    }
}
