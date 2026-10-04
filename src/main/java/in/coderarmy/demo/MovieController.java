package in.coderarmy.demo;

import in.coderarmy.demo.model.MovieMatch;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/movies")
public class MovieController {

    private final MovieService movieService;

    public MovieController(MovieService movieService) {
        this.movieService = movieService;
    }

    @GetMapping("/search")
    public List<MovieMatch> search(@RequestParam String query) {
        return movieService.search(query);
    }

    @GetMapping("/{title}/similar")
    public List<MovieMatch> similarMovies(@PathVariable String title) {
        return movieService.similarMovies(title);
    }

    @GetMapping("/dimensions")
    public Map<String, Object> getDimensions() {
        return Map.of(
                "totalMovies", movieService.getMoviesCount(),
                "dimensions", movieService.getEmbeddingDimensions()
        );
    }
}
