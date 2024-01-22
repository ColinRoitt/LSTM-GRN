import pygame
import math
import random
import numpy as np
from ELSTM import ELSTM_Dynamic as ELSTM
# from ELSTM_v2 import ELSTM

angleoffset = 90
size = width, height = 1000, 1000  # display with/height
startpos = width / 2, height / 2

ORANGE = (227, 138, 43)
BLUE = (56, 76, 107)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def polar_to_cart(theta, r, pos):
    offx, offy = pos
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return tuple([x + y for x, y in zip((int(x), int(y)), (offx, offy))])

def generate_random_points(min_radius, max_radius, num_points):
    points = []
    
    for _ in range(num_points):
        # Generate a random radius between min_radius and max_radius
        radius = random.uniform(min_radius, max_radius)
        
        # Generate a random angle between 0 and 2*pi (360 degrees)
        angle = random.uniform(0, 2 * math.pi)
        
        # Convert polar coordinates to Cartesian coordinates
        x = radius * math.cos(angle) + width / 2
        y = radius * math.sin(angle) + height / 2
        
        points.append((x, y))
    
    return points

def drawTree(oldpos, genome = [], units = 5, targets = None, hide_display = False, fast = False, save_devo_images = False, devo_save_output = None):
    # genome = np.random.rand(ELSTM.get_genome_size(units))
    if not fast:
        pygame.init()  # init display
        # set background color to white
        if hide_display:
            screen = pygame.display.set_mode(size, flags=pygame.HIDDEN)  # open screen
            screen.fill(WHITE)
        else:
            screen = pygame.display.set_mode(size)
            screen.fill(WHITE)
    angle = 0  # angle
    a = 0  # angle
    step = 0  # step size / line length
    newpos = oldpos
    num = []  # stack for the brackets

    color = BLACK
    linesize = 1
    i = 0
    running = True
    moves = []
    # closeness = [0 for _ in range(targets.shape[0])]
    non_caputuring_movements = 0

    MAX_DEVO = 5000

    if targets is None:
        # targets = np.array([(450, 450), (width / 2, 255), (width / 2, height - 225)])
        targets = np.array(generate_random_points(200, 500, 100))
    target_radius = 10
    found_targets = dict.fromkeys([tuple(t) for t in targets], False)

    # look_ahead = 200
    look_ahead = 1000

    grn = ELSTM(units, genome, continuous = True, outputs = units)

    if not fast:
        # draw targets
        for t in targets:
            pygame.draw.circle(screen, BLUE, t, target_radius, 0)

    while running:

        if save_devo_images & (i % 249 == 0):
            pygame.image.save(screen, devo_save_output + str(i) + ".png")

        if i >= MAX_DEVO:
            running = False
            if save_devo_images:
                pygame.image.save(screen, devo_save_output + str(i) + ".png")
            break
        i += 1
        
        # get inputs for GRN
        x = oldpos[0]
        y = oldpos[1]
        # step agiven
        # angle agiven
        # caluclate closest target to oldpos with max look ahead of 30
        explored_point = np.array(oldpos)
        distances = np.linalg.norm(targets[:, :2] - explored_point, axis=1)
        if np.min(distances) > look_ahead:
            closest_index = np.argmin(distances)
            vector_to_closest_target = np.array(targets[closest_index, :2]) - explored_point
        else:
            vector_to_closest_target = np.array([0, 0])
        
        targets_left = np.sum(np.logical_not(list(found_targets.values())))

        input_data = np.array([x, y, step, angle, vector_to_closest_target[0], vector_to_closest_target[1]])
        # input_data = np.array([x, y, step, angle, vector_to_closest_target[0], vector_to_closest_target[1], targets_left])
        input_data = input_data.flatten()
        # normalize
        input_data = input_data / 1000 
        output = grn.forward(np.array(input_data, dtype=np.float32))
        print(output)
        step, angle, terminate, pop, push, _ = output.T
        print(terminate)
        # step, angle, terminate, pop, push, _, _ = output

        step = abs(step * 100)
        angle = angle * 180

        if terminate < 0:
            running = False
        if pop > 0 and len(num) > 0:
                oldpos, a = num.pop()
        if push > 0:
            num.append((oldpos, a))

        moves.append((step, angle, oldpos))

        a += angle
        newpos = polar_to_cart(a + angleoffset, step, oldpos)
        if not fast:
            pygame.draw.line(screen, color, oldpos, newpos, linesize)

        if not fast:
            pygame.display.flip()
        
        intersected_indices = []
        movement_vector = np.array(newpos) - np.array(oldpos)
        movement_magnitude_squared = np.dot(movement_vector, movement_vector)
        if movement_magnitude_squared > 0:
            target_coordinates = targets[:, :2]
            target_vectors = target_coordinates - np.array(oldpos)
            dot_products = np.dot(target_vectors, movement_vector)
            target_distances_squared = np.sum(target_vectors**2, axis=1)
            # Calculate the discriminant for all targets
            discriminants = dot_products**2 - movement_magnitude_squared * (target_distances_squared - target_radius**2)
            # Find the indices of targets with non-negative discriminants and within the movement range
            intersected_indices = np.where((discriminants >= 0) & (dot_products >= 0) & (dot_products <= movement_magnitude_squared))[0]

            for idx in intersected_indices:
                target_x, target_y = targets[idx]
                found_targets[(target_x, target_y)] = True

        # if len(intersected_indices) == 0:
        #     non_caputuring_movements += 1
        # else:
        #     for idx in intersected_indices:
        #         target_x, target_y = targets[idx]
        #         if not found_targets[(target_x, target_y)]:
        #             non_caputuring_movements += step
        #             break

        # explored_point = np.array(newpos)
        # distances = np.linalg.norm(targets[:, :2] - explored_point, axis=1)
        # target_indices = np.where(distances < target_radius)
        # targets_within_radius = targets[target_indices]
        # for t_found in targets_within_radius:
        #     found_targets[tuple(t_found)] = True
            # print("TARGET FOUND, ", t_found)

        # for i in range(len(targets)):
        #     distances = np.linalg.norm(targets[i] - explored_point)
        #     # map distance to 0-1
        #     current_closeness = 1 - (distances / 1000)
        #     if current_closeness > closeness[i]:
        #         closeness[i] = current_closeness

        if not fast:
            for t, found in found_targets.items():
                if found:
                    pygame.draw.circle(screen, ORANGE, t, target_radius, 0)

        oldpos = newpos

    if not fast:
        score = sum(found_targets.values())
        # score = sum(closeness)
        print("SCORE: ", score)

        pygame.display.flip()
        # pygame.image.save(screen, "screenshot.png")
        print("Finished")
        waiting = True
        if not hide_display:
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # pygame.quit()
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        # pygame.quit()
                        waiting = False

        return screen, sum(found_targets.values()), i, moves
    return sum(found_targets.values()), i
    # return sum(closeness), i

def fitness_function(genome, units, targets = None):
    score, steps = drawTree(startpos, genome, units, hide_display=True, fast=True, targets=targets)
    return score, steps

def display(genome, units):
    return drawTree(startpos, genome, units)

def draw_and_save(genome, units, output, targets = None, output_folder = None, index = 0):
    screen, score, i, moves = drawTree(startpos, genome, units, hide_display = True, targets=targets, save_devo_images=True, devo_save_output = f"{output_folder}devo/{index}-")
    # write score in red text top right
    font = pygame.font.Font('freesansbold.ttf', 21)
    text = font.render(f'score: {str(score)} - steps: {str(i)}', True, ORANGE, WHITE)
    textRect = text.get_rect()
    textRect.topright = (width - 10, 10)
    screen.blit(text, textRect)
    pygame.image.save(screen, output)
    return score, i, moves

def draw_and_save_in_devo(genome, units, output, targets = None):
    _, score, i, moves = drawTree(startpos, genome, units, hide_display = True, targets=targets, save_devo_images=True, devo_save_output = output)
    return score, i, moves

if __name__ == '__main__':
    print('Drawing structure...')
    drawTree(startpos, genome=np.random.rand(ELSTM.get_genome_size(5)), units=5)
        
    