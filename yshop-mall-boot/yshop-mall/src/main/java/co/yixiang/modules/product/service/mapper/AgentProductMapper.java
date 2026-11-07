package co.yixiang.modules.product.service.mapper;

import co.yixiang.modules.product.service.dto.AgentProductDto;
import co.yixiang.modules.product.service.dto.AgentProductSearchQuery;
import co.yixiang.modules.product.service.dto.AgentProductSkuDto;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.springframework.stereotype.Repository;

import java.util.List;

/** All statements are SELECTs, with the same SKU predicate for eligibility and returned rows. */
@Repository
public interface AgentProductMapper {
    String SKU_FILTER = """
            s.stock > 0 AND s.price >= 0 AND s.`unique` IS NOT NULL
            AND CHAR_LENGTH(TRIM(s.`unique`)) > 0
            <if test="query.minPrice != null">AND s.price &gt;= #{query.minPrice}</if>
            <if test="query.maxPrice != null">AND s.price &lt;= #{query.maxPrice}</if>
            """;

    String PRODUCT_FILTER = """
            p.is_show = 1 AND p.is_del = 0 AND p.is_integral = 0
            <if test="query.categoryId != null">
              AND FIND_IN_SET(CAST(#{query.categoryId} AS CHAR), p.cate_id) > 0
            </if>
            <if test="query.keywordPattern != null">
              AND (p.store_name LIKE #{query.keywordPattern} ESCAPE '!'
                   OR p.store_info LIKE #{query.keywordPattern} ESCAPE '!'
                   OR p.keyword LIKE #{query.keywordPattern} ESCAPE '!')
            </if>
            AND EXISTS (SELECT 1 FROM yx_store_product_attr_value s WHERE s.product_id = p.id AND
            """ + SKU_FILTER + ") ";

    @Select("<script>SELECT COUNT(*) FROM yx_store_product p WHERE " + PRODUCT_FILTER + "</script>")
    long countProducts(@Param("query") AgentProductSearchQuery query);

    @Select("""
            <script>
            SELECT p.id AS product_id, p.store_name AS name, COALESCE(p.store_info, '') AS description,
                   COALESCE(p.image, '') AS image
            FROM yx_store_product p WHERE
            """ + PRODUCT_FILTER + " ORDER BY p.id ASC LIMIT #{query.limit} OFFSET #{query.offset}</script>")
    List<AgentProductDto> selectProducts(@Param("query") AgentProductSearchQuery query);

    @Select("""
            <script>
            SELECT s.product_id, s.sku AS name, s.`unique` AS unique_id, s.price, s.stock,
                   COALESCE(s.image, '') AS image
            FROM yx_store_product_attr_value s WHERE
            """ + SKU_FILTER + """
            AND s.product_id IN
            <foreach collection="productIds" item="productId" open="(" separator="," close=")">
              #{productId}
            </foreach>
            ORDER BY s.product_id ASC, s.id ASC
            </script>
            """)
    List<AgentProductSkuDto> selectSkus(@Param("query") AgentProductSearchQuery query,
                                      @Param("productIds") List<Long> productIds);
}
