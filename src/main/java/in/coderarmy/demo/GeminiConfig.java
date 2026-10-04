package in.coderarmy.demo;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Configuration
public class GeminiConfig {

    @Value("${google.gemini.api-key}")
    private String apiKey;

    @Bean
    public EmbeddingModel embeddingModel(ObjectMapper objectMapper) {
        RestTemplate restTemplate = new RestTemplate();
        // Google ka current active production model: gemini-embedding-001
        String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=" + apiKey;

        return new EmbeddingModel() {
            @Override
            public float[] embed(String text) {
                try {
                    Map<String, Object> part = Map.of("text", text);
                    Map<String, Object> content = Map.of("parts", List.of(part));
                    Map<String, Object> body = Map.of("content", content);

                    HttpHeaders headers = new HttpHeaders();
                    headers.setContentType(MediaType.APPLICATION_JSON);
                    HttpEntity<Map<String, Object>> entity = new HttpEntity<>(body, headers);

                    ResponseEntity<String> response = restTemplate.postForEntity(url, entity, String.class);
                    JsonNode root = objectMapper.readTree(response.getBody());
                    JsonNode valuesNode = root.path("embedding").path("values");

                    float[] embedding = new float[valuesNode.size()];
                    for (int i = 0; i < valuesNode.size(); i++) {
                        embedding[i] = (float) valuesNode.get(i).asDouble();
                    }
                    return embedding;
                } catch (Exception e) {
                    throw new RuntimeException("Gemini embedding call failed: " + e.getMessage(), e);
                }
            }

            @Override
            public float[] embed(Document document) {
                return embed(document.getContent());
            }

            @Override
            public EmbeddingResponse call(EmbeddingRequest request) {
                throw new UnsupportedOperationException();
            }

            @Override
            public List<float[]> embed(List<String> texts) {
                List<float[]> list = new ArrayList<>();
                for (String t : texts) {
                    list.add(embed(t));
                }
                return list;
            }

            @Override
            public int dimensions() {
                return 768;
            }
        };
    }
}