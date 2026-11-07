package co.yixiang.modules.product;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.ibatis.mapping.Environment;
import org.apache.ibatis.session.Configuration;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;
import org.apache.ibatis.transaction.jdbc.JdbcTransactionFactory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationTargetException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Exercises the production MyBatis statements against MySQL session-only fixture tables. */
class AgentProductSearchTest {
    private Class<?> queryType;
    private Object service;
    private SqlSession session;
    private Connection connection;
    private final ObjectMapper json = new ObjectMapper();

    @BeforeEach
    @SuppressWarnings({"rawtypes", "unchecked"})
    void fixtures() throws Exception {
        queryType = assertDoesNotThrow(() -> Class.forName(
                "co.yixiang.modules.product.service.dto.AgentProductSearchQuery"),
                "The agent product search query is not implemented");
        Class mapperType = Class.forName("co.yixiang.modules.product.service.mapper.AgentProductMapper");
        Class<?> serviceType = Class.forName("co.yixiang.modules.product.service.AgentProductSearchService");
        Map<String, String> env = new HashMap<>(System.getenv());
        Path envFile = Path.of(System.getProperty("agent.test.env", "../../yshop-agent/.env"));
        if (Files.exists(envFile)) {
            for (String line : Files.readAllLines(envFile)) {
                int equals = line.indexOf('=');
                if (equals > 0 && !line.stripLeading().startsWith("#")) {
                    String value = line.substring(equals + 1).strip();
                    if (value.length() >= 2 && ((value.startsWith("\"") && value.endsWith("\""))
                            || (value.startsWith("'") && value.endsWith("'")))) {
                        value = value.substring(1, value.length() - 1);
                    }
                    env.putIfAbsent(line.substring(0, equals).strip(), value);
                }
            }
        }
        assertTrue(env.containsKey("YSHOP_DB_PASSWORD"), "Local MySQL fixture credentials are required");
        String dbName = env.get("YSHOP_DB_NAME");
        assertNotNull(dbName, "Local MySQL fixture database name is required");
        assertTrue(dbName.matches("[A-Za-z0-9_]+"), "Invalid fixture database name");
        connection = DriverManager.getConnection("jdbc:mysql://127.0.0.1:3306/" + dbName
                + "?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=Asia/Shanghai",
                env.get("YSHOP_DB_USER"), env.get("YSHOP_DB_PASSWORD"));
        try (Statement statement = connection.createStatement()) {
            statement.execute("CREATE TEMPORARY TABLE yx_store_product (id BIGINT PRIMARY KEY, "
                    + "store_name VARCHAR(256), store_info VARCHAR(512), keyword VARCHAR(256), "
                    + "image VARCHAR(256), cate_id VARCHAR(256), price DECIMAL(12,2), "
                    + "is_show INT, is_del INT, is_integral INT)");
            statement.execute("CREATE TEMPORARY TABLE yx_store_product_attr_value (id BIGINT PRIMARY KEY, "
                    + "product_id BIGINT, sku VARCHAR(128), `unique` VARCHAR(100), "
                    + "price DECIMAL(12,2), stock INT, image VARCHAR(256))");
        }
        for (int id = 1; id <= 12; id++) {
            try (PreparedStatement statement = connection.prepareStatement(
                    "INSERT INTO yx_store_product VALUES (?, ?, ?, ?, ?, ?, 99.00, ?, ?, ?)")) {
                statement.setInt(1, id);
                statement.setString(2, id == 1 ? "50Z normal" : "product " + id);
                statement.setString(3, id == 3 ? "Literal 50% _ deal" : "description");
                statement.setString(4, id == 3 ? "needle!key" : "ordinary");
                statement.setString(5, "product.png");
                statement.setString(6, id == 2 ? "11" : id == 3 ? "1,11" : "1");
                statement.setInt(7, id == 5 ? 0 : 1);
                statement.setInt(8, id == 6 ? 1 : 0);
                statement.setInt(9, id == 7 ? 1 : 0);
                statement.executeUpdate();
            }
        }
        sku(1, 1, "ten", "10.00", 3, "ten");
        sku(2, 1, "twenty", "20.00", 2, "twenty");
        sku(3, 1, "zero", "0.00", 2, "zero");
        sku(4, 1, "depleted", "10.00", 0, "depleted");
        sku(5, 1, "bad-price", "-1.00", 3, "negative");
        sku(6, 1, "", "5.00", 3, "blank");
        sku(7, 2, "eleven", "10.00", 1, "default");
        sku(8, 3, "fifteen", "15.00", 1, "default");
        sku(9, 4, "out", "10.00", 0, "default");
        sku(10, 5, "hidden", "10.00", 1, "default");
        sku(11, 6, "deleted", "10.00", 1, "default");
        sku(12, 7, "integral", "10.00", 1, "default");
        sku(13, 8, "negative", "-1.00", 1, "default");
        sku(14, 9, "   ", "10.00", 1, "default");
        sku(15, 10, null, "10.00", 1, "default");
        sku(16, 11, "free", "0.00", 1, "default");
        Configuration config = new Configuration();
        config.setMapUnderscoreToCamelCase(true);
        config.setEnvironment(new Environment("temporary-fixtures", new JdbcTransactionFactory(),
                new org.springframework.jdbc.datasource.SingleConnectionDataSource(connection, true)));
        config.addMapper(mapperType);
        session = new SqlSessionFactoryBuilder().build(config).openSession(connection);
        service = serviceType.getConstructor(mapperType).newInstance(session.getMapper(mapperType));
    }

    private void sku(int id, int product, String unique, String price, int stock, String name) throws Exception {
        try (PreparedStatement statement = connection.prepareStatement(
                "INSERT INTO yx_store_product_attr_value VALUES (?, ?, ?, ?, ?, ?, ?)")) {
            statement.setInt(1, id);
            statement.setInt(2, product);
            statement.setString(3, name);
            statement.setString(4, unique);
            statement.setBigDecimal(5, new BigDecimal(price));
            statement.setInt(6, stock);
            statement.setString(7, "sku.png");
            statement.executeUpdate();
        }
    }

    @AfterEach
    void closeFixtures() throws Exception {
        if (session != null) session.close();
        if (connection != null && !connection.isClosed()) connection.close();
    }

    private Object query(String keyword, Long category, String minimum, String maximum, int page, int limit)
            throws Exception {
        return queryType.getConstructor(String.class, Long.class, BigDecimal.class, BigDecimal.class,
                int.class, int.class).newInstance(keyword, category,
                minimum == null ? null : new BigDecimal(minimum),
                maximum == null ? null : new BigDecimal(maximum), page, limit);
    }

    private JsonNode search(String keyword, Long category, String minimum, String maximum, int page, int limit)
            throws Exception {
        Object result = service.getClass().getMethod("search", queryType)
                .invoke(service, query(keyword, category, minimum, maximum, page, limit));
        return json.readTree(json.writeValueAsBytes(result));
    }

    private List<Long> ids(JsonNode result) {
        List<Long> ids = new ArrayList<>();
        result.path("products").forEach(product -> ids.add(product.path("product_id").asLong()));
        return ids;
    }

    @Test void equalBoundsUseSkuPriceAndRemoveOutOfBudgetAndDepletedSkus() throws Exception {
        JsonNode result = search("", null, "10.00", "10.00", 1, 10);
        assertEquals(List.of(1L, 2L), ids(result));
        assertEquals(2, result.path("total").asInt());
        JsonNode product = result.path("products").get(0);
        assertEquals(1, product.path("skus").size());
        assertEquals("ten", product.path("skus").get(0).path("unique_id").asText());
        assertEquals("10.00", product.path("skus").get(0).path("price").asText());
        assertTrue(product.path("skus").get(0).path("price").isTextual());
        assertEquals(5, product.size());
        assertEquals(5, product.path("skus").get(0).size());
        assertEquals(4, result.size());
    }

    @Test void upperBoundIsInclusiveAndIncludesFreeSkus() throws Exception {
        assertEquals(List.of(1L, 2L, 11L), ids(search("", null, null, "10", 1, 10)));
        assertEquals(List.of(1L, 11L), ids(search("", null, "0", "0", 1, 10)));
    }

    @Test void lowerBoundIsInclusiveAndDoesNotUseProductStartingPrice() throws Exception {
        JsonNode result = search("", null, "15", null, 1, 10);
        assertEquals(List.of(1L, 3L), ids(result));
        assertEquals("twenty", result.path("products").get(0).path("skus").get(0).path("unique_id").asText());
        assertEquals(List.of(), ids(search("", null, "98", "100", 1, 10)));
    }

    @Test void returnsOnlyPurchasableOrdinaryProducts() throws Exception {
        assertEquals(List.of(1L, 2L, 3L, 11L), ids(search("", null, null, null, 1, 10)));
    }

    @Test void categoryUsesExactCommaSeparatedIds() throws Exception {
        assertEquals(List.of(1L, 3L, 11L), ids(search("", 1L, null, null, 1, 10)));
        assertEquals(List.of(2L, 3L), ids(search("", 11L, null, null, 1, 10)));
        assertEquals(List.of(), ids(search("", Long.MAX_VALUE, null, null, 1, 10)));
    }

    @Test void keywordTreatsPercentUnderscoreAndEscapeCharacterLiterally() throws Exception {
        for (String keyword : List.of("50%", "_", "!", "  needle  ", "Literal")) {
            assertEquals(List.of(3L), ids(search(keyword, null, null, null, 1, 10)), keyword);
        }
        assertEquals(List.of(1L), ids(search("50Z", null, null, null, 1, 10)));
        assertEquals(List.of(), ids(search("' OR 1=1 --", null, null, null, 1, 10)));
    }

    @Test void paginationCountsProductsAndKeepsTheirCompleteMatchingSkuList() throws Exception {
        JsonNode first = search("", null, null, null, 1, 1);
        assertEquals(List.of(1L), ids(first));
        assertEquals(3, first.path("products").get(0).path("skus").size());
        assertEquals(4, first.path("total").asInt());
        JsonNode second = search("", null, null, null, 2, 1);
        assertEquals(List.of(2L), ids(second));
        assertEquals(2, second.path("page").asInt());
        assertEquals(1, second.path("limit").asInt());
        assertEquals(4, second.path("total").asInt());
    }

    @Test void beyondLastPageReturnsEmptyProductsAndRetainsTotal() throws Exception {
        JsonNode result = search("", null, null, null, Integer.MAX_VALUE, 20);
        assertEquals(List.of(), ids(result));
        assertEquals(4, result.path("total").asInt());
    }

    @Test void invalidQueryParametersFailBeforeSql() throws Exception {
        assertInvalid("x".repeat(101), null, null, null, 1, 10);
        assertInvalid("", 0L, null, null, 1, 10);
        assertInvalid("", -1L, null, null, 1, 10);
        assertInvalid("", null, "-0.01", null, 1, 10);
        assertInvalid("", null, null, "-0.01", 1, 10);
        assertInvalid("", null, "10.01", "10", 1, 10);
        assertInvalid("", null, null, null, 0, 10);
        assertInvalid("", null, null, null, -1, 10);
        assertInvalid("", null, null, null, 1, 0);
        assertInvalid("", null, null, null, 1, 21);
        assertNotNull(query("  " + "x".repeat(100) + "  ", 1L, "10", "10.00", 1, 20));
    }

    private void assertInvalid(String keyword, Long category, String minimum, String maximum, int page, int limit) {
        InvocationTargetException error = assertThrows(InvocationTargetException.class,
                () -> query(keyword, category, minimum, maximum, page, limit));
        assertInstanceOf(IllegalArgumentException.class, error.getCause());
    }
}
