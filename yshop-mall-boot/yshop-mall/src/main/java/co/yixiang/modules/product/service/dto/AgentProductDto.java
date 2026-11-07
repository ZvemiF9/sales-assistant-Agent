package co.yixiang.modules.product.service.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/** Public search product whitelist; it contains no merchant or cost fields. */
@Getter
@Setter
public class AgentProductDto {
    @JsonProperty("product_id")
    private Long productId;
    private String name;
    private String description;
    private String image;
    private List<AgentProductSkuDto> skus = new ArrayList<>();
}
