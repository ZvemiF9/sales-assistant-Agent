package co.yixiang.modules.product.rest;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

class AgentProductControllerTest {
    private MockMvc mvc;
    private AtomicInteger queries;

    @BeforeEach void controller() throws Exception {
        Class<?> controllerType = assertDoesNotThrow(() -> Class.forName(
                "co.yixiang.modules.product.rest.AgentProductController"),
                "The agent products GET endpoint is not implemented");
        Class<?> mapperType = Class.forName("co.yixiang.modules.product.service.mapper.AgentProductMapper");
        Class<?> serviceType = Class.forName("co.yixiang.modules.product.service.AgentProductSearchService");
        queries = new AtomicInteger();
        Object mapper = Proxy.newProxyInstance(mapperType.getClassLoader(), new Class<?>[]{mapperType},
                (proxy, method, arguments) -> {
                    queries.incrementAndGet();
                    return method.getReturnType() == long.class ? 0L : List.of();
                });
        Object service = serviceType.getConstructor(mapperType).newInstance(mapper);
        Object controller = controllerType.getConstructor(serviceType).newInstance(service);
        mvc = MockMvcBuilders.standaloneSetup(controller).build();
    }

    @Test void defaultRequestUsesWhitelistedSnakeCaseContract() throws Exception {
        mvc.perform(get("/agent/products"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.data.products").isEmpty())
                .andExpect(jsonPath("$.data.page").value(1))
                .andExpect(jsonPath("$.data.limit").value(10))
                .andExpect(jsonPath("$.data.total").value(0));
        assertEquals(1, queries.get());
    }

    @Test void bindsSnakeCaseDecimalsAndPagination() throws Exception {
        mvc.perform(get("/agent/products").param("keyword", " hello ").param("category_id", "1")
                .param("min_price", "10.01").param("max_price", "10.010").param("page", "2").param("limit", "20"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.data.page").value(2))
                .andExpect(jsonPath("$.data.limit").value(20));
    }

    @Test void invalidInputsReturnParameterErrorWithoutExecutingSql() throws Exception {
        String[][] invalid = {{"page", "0"}, {"page", "1.5"}, {"page", "2147483648"},
                {"limit", "21"}, {"limit", "-1"}, {"category_id", "0"}, {"category_id", "1.5"},
                {"category_id", "9223372036854775808"}, {"min_price", "NaN"}, {"max_price", "Infinity"},
                {"min_price", "-0.01"}, {"max_price", "-1"}, {"keyword", "x".repeat(101)}};
        for (String[] pair : invalid) {
            mvc.perform(get("/agent/products").param(pair[0], pair[1]))
                    .andExpect(status().isBadRequest()).andExpect(jsonPath("$.success").value(false));
        }
        mvc.perform(get("/agent/products").param("min_price", "2").param("max_price", "1"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.success").value(false));
        assertEquals(0, queries.get());
    }
}
