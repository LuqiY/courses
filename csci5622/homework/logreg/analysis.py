from logreg import *
import numpy


def question1():
    etas = {
        .01: [],
        .1: [],
        1: [],
        10: [],
        100: []}

    passes = 10

    train, test, vocab = read_dataset("../data/autos_motorcycles/positive",
                                      "../data/autos_motorcycles/negative", "../data/autos_motorcycles/vocab")

    for eta in etas:
        print eta
        # Initialize model
        lr = LogReg(len(vocab), 0.0, lambda x: eta)

        # Iterations
        iteration = 0
        for pp in xrange(passes):
            random.shuffle(train)
            for ex in train:
                lr.sg_update(ex, iteration)
                if iteration % 1000 == 1:
                    # train_lp, train_acc = lr.progress(train)
                    ho_lp, ho_acc = lr.progress(test)
                    etas[eta].append(ho_lp)
                    # print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                    # (iteration, train_lp, ho_lp, train_acc, ho_acc))
                iteration += 1

    for eta in sorted(etas.iterkeys()):
        y = etas[eta]
        x = range(0, len(y))
        plt.plot(x, y, label=str(eta))
    plt.legend(loc='lower right')
    plt.title('Convergence of log-likelihood')
    plt.ylabel('Hold-out Set log-likelihood')
    plt.xlabel('Iterations')
    plt.savefig('q1.png')

question1()