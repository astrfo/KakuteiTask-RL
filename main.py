from simulator import Simulator

def main():
    trial = 100 #シミュレーション数
    episode = 10000
    sim = Simulator(trial, episode)
    sim.run()

if __name__ == '__main__':
    print('started run')
    main()
    print('finished run')