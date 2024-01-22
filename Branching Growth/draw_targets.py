import pygame
import sys
import os

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
DOT_RADIUS = 10
RED = (255, 0, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Red Dot Drawer")

# List to store dot locations
dot_locations = []

running = True
drawing = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button click
                x, y = pygame.mouse.get_pos()
                dot_locations.append((x, y))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Ask the user for a name
                name = input("Enter a name for the shape: ")
                
                # Create the shapes folder if it doesn't exist
                shapes_folder = "shapes"
                if not os.path.exists(shapes_folder):
                    os.mkdir(shapes_folder)
                
                # Create a Python file with the given name and save dot_locations
                filename = os.path.join(shapes_folder, name + ".py")
                with open(filename, "w") as file:
                    file.write(f"dot_locations = {dot_locations}")

                print(f"Shape saved as {filename}")
                
                # Reset dot_locations for a new drawing
                dot_locations = []

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw red dots
    for location in dot_locations:
        pygame.draw.circle(screen, RED, location, DOT_RADIUS)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
