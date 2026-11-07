package co.yixiang.modules.product.service.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import lombok.Getter;
import lombok.Setter;

import java.math.BigDecimal;

/** Public matching-SKU whitelist, with an internal product association for grouping. */
@Getter
@Setter
public class AgentProductSkuDto {
    @JsonIgnore
    private Long productId;
    private String name;
    @JsonProperty("unique_id")
    private String uniqueId;
    @JsonSerialize(using = ToStringSerializer.class)
    private BigDecimal price;
    private Integer stock;
    private String image;
}
