try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
