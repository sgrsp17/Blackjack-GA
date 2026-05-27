try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class EvolutionDisplay:
    """Pygame window that shows live evolution progress during training."""

    _W, _H = 620, 420
    _BG = (15, 15, 25)

    def __init__(self, total_generations, selection_method):
        import pygame
        pygame.init()
        self._pg = pygame
        self.screen = pygame.display.set_mode((self._W, self._H))
        pygame.display.set_caption(f"GA Evolution — {selection_method} selection")
        self._total = total_generations
        self._method = selection_method
        self._f_large = pygame.font.SysFont("Arial", 30, bold=True)
        self._f_med = pygame.font.SysFont("Arial", 21)
        self._max_hist = []
        self._avg_hist = []

    def update(self, generation, max_fitness, avg_fitness, elapsed):
        self._max_hist.append(max_fitness)
        self._avg_hist.append(avg_fitness)

        for event in self._pg.event.get():
            pass  # keep window responsive

        self.screen.fill(self._BG)

        title = self._f_large.render("EVOLUTION IN PROGRESS", True, (200, 200, 255))
        self.screen.blit(title, (self._W // 2 - title.get_width() // 2, 18))

        sel = self._f_med.render(f"Selection: {self._method}", True, (160, 160, 160))
        self.screen.blit(sel, (self._W // 2 - sel.get_width() // 2, 56))

        gen_txt = self._f_large.render(
            f"Generation  {generation} / {self._total}", True, (255, 255, 255)
        )
        self.screen.blit(gen_txt, (self._W // 2 - gen_txt.get_width() // 2, 92))

        # progress bar
        bx, by, bw, bh = 50, 138, self._W - 100, 18
        self._pg.draw.rect(self.screen, (40, 40, 70), (bx, by, bw, bh))
        fill = int(bw * generation / self._total)
        self._pg.draw.rect(self.screen, (70, 150, 255), (bx, by, fill, bh))
        self._pg.draw.rect(self.screen, (130, 130, 190), (bx, by, bw, bh), 2)

        max_col = (80, 240, 80) if max_fitness >= 0 else (240, 80, 80)
        self.screen.blit(
            self._f_med.render(f"Best Fitness:  {max_fitness:.3f}", True, max_col), (50, 172)
        )
        self.screen.blit(
            self._f_med.render(f"Avg  Fitness:  {avg_fitness:.3f}", True, (220, 200, 80)), (50, 200)
        )

        m, s = int(elapsed // 60), int(elapsed % 60)
        self.screen.blit(
            self._f_med.render(f"Elapsed:  {m:02d}:{s:02d}", True, (160, 160, 160)), (50, 232)
        )

        if len(self._max_hist) > 1:
            self._draw_graph()

        self._pg.display.flip()

    def _draw_graph(self):
        gx, gy, gw, gh = 50, 272, self._W - 100, 120
        self._pg.draw.rect(self.screen, (28, 28, 48), (gx, gy, gw, gh))
        self._pg.draw.rect(self.screen, (70, 70, 110), (gx, gy, gw, gh), 1)

        all_v = self._max_hist + self._avg_hist
        lo, hi = min(all_v), max(all_v)
        span = hi - lo if hi != lo else 1.0
        n = len(self._max_hist)

        def px(val, idx):
            x = gx + int(idx / max(n - 1, 1) * gw)
            y = gy + gh - int((val - lo) / span * gh)
            return (x, max(gy, min(gy + gh, y)))

        for hist, col in [(self._max_hist, (80, 220, 80)), (self._avg_hist, (220, 200, 60))]:
            pts = [px(v, i) for i, v in enumerate(hist)]
            self._pg.draw.lines(self.screen, col, False, pts, 2)

        leg_y = gy + gh + 6
        self._pg.draw.line(self.screen, (80, 220, 80), (gx, leg_y + 7), (gx + 20, leg_y + 7), 2)
        self.screen.blit(
            self._pg.font.SysFont("Arial", 14).render("Max", True, (80, 220, 80)), (gx + 24, leg_y)
        )
        self._pg.draw.line(self.screen, (220, 200, 60), (gx + 70, leg_y + 7), (gx + 90, leg_y + 7), 2)
        self.screen.blit(
            self._pg.font.SysFont("Arial", 14).render("Avg", True, (220, 200, 60)), (gx + 94, leg_y)
        )

    def close(self):
        self._pg.quit()


def draw_game_summary(screen, game_results):
    import pygame

    sidebar_width = 220
    height = screen.get_height()
    original_width = screen.get_width()

    # If we haven't expanded the window yet, make the display wider and keep the game on the left
    if not getattr(draw_game_summary, "expanded", False):
        content_copy = screen.copy()
        new_width = original_width + sidebar_width + 20
        new_screen = pygame.display.set_mode((new_width, height))
        new_screen.blit(content_copy, (0, 0))
        draw_game_summary.expanded = True
        draw_game_summary.game_width = original_width
        draw_game_summary.screen = new_screen
        screen = new_screen
    else:
        screen = pygame.display.get_surface()
        original_width = draw_game_summary.game_width

    sidebar = pygame.Surface((sidebar_width, height))
    sidebar.set_alpha(230)
    sidebar.fill((20, 20, 20))
    pygame.draw.rect(sidebar, (200, 200, 200), sidebar.get_rect(), 2)

    font_title = pygame.font.SysFont('Arial', 24, bold=True)
    font_text = pygame.font.SysFont('Arial', 20)

    title_surface = font_title.render('JOGOS', True, (255, 255, 255))
    sidebar.blit(title_surface, (10, 10))

    start_index = max(0, len(game_results) - 10)
    for idx, result in enumerate(game_results[start_index:], start_index + 1):
        line = f'Jogo {idx}: {result}'
        color = (255, 255, 255)
        if result == 'W':
            color = (0, 255, 0)
        elif result == 'L':
            color = (255, 0, 0)
        elif result == 'D':
            color = (255, 255, 0)
        text_surface = font_text.render(line, True, color)
        sidebar.blit(text_surface, (10, 40 + 26 * (idx - start_index - 1)))

    screen.blit(sidebar, (original_width + 10, 10))
    pygame.display.update()


def plot_fitness_history(max_history, avg_history):
    if not HAS_MATPLOTLIB:
        print("\n[Warning] matplotlib is not installed. Install it with `pip install matplotlib` to see the fitness graph.")
        return

    generations = list(range(1, len(max_history) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_history, label='Max Fitness', marker='o')
    plt.plot(generations, avg_history, label='Average Fitness', marker='o')
    plt.title('Fitness Evolution Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fitness_evolution.png')
    print("Fitness graph saved to 'fitness_evolution.png'.")
    plt.show()
