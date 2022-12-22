#Takes in a tensor nx3
#Develop a reprentation of the context
def cost_func(solution):

    x1 = solution[:,0]
    x2 = solution[:,1]

    reward = 2*x1 + x2*x2

    #reward = solution*solution - solution

    return reward
