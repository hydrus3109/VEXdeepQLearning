def render_environment(state, robot_position, rings, goals):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.xlim(0, state['field_width'])
    plt.ylim(0, state['field_height'])

    # Draw the robot
    plt.scatter(robot_position[0], robot_position[1], c='blue', s=100, label='Robot')

    # Draw the rings
    for ring in rings:
        plt.scatter(ring['position'][0], ring['position'][1], c='orange', s=50, label='Ring')

    # Draw the goals
    for goal in goals:
        plt.scatter(goal['position'][0], goal['position'][1], c='green', s=200, label='Goal')

    plt.title('VEX High Stakes Environment')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.legend()
    plt.grid()
    plt.show()