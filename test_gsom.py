import unittest
import Refactored_gsom
import numpy as np
from Refactored_gsom import neuron

__author__ = 'laksheen'


class TestGsom(unittest.TestCase):

    def setUp(self):
        print 'in setup'
        self.map = Refactored_gsom.gsom(SP=0.9999,dims=16,nr_s=10,lr_s=0.01,lrr=0.95,fd=0.999)
        #self.map._viewmap()

        data = np.loadtxt("zoo.data.txt",dtype=str,delimiter=",")
        data = np.array(data)
        names = data[:,0]
        names= np.column_stack((names,data[:,-1]))
        self.features= data[:,:-1]
        #print features
        self.features = self.features[:,1:].astype(int)

    def tearDown(self):
        print 'in tear down'
        del self.map
        del self.features

    '''
    def test__initial_neurons(self):

        print 'in test 1'
        print 'number of neurons ',len(self.map.map_neurons)
        list= []
        count=0
        for neu in self.map.map_neurons.values():
            nhash = str(neu.x_c)+" "+str(neu.y_c)
            print neu.weight_vs;
            count += 1
            if len(nhash) != 0:
                list.append(True)
            else:
                list.append(False)

        self.assertTrue(np.all(list))
        self.assertEqual(4,count,'4 neurons not generated')
    '''

    def test__predict_point(self):
        print 'in test 2'
        output = self.map._predict_point(self.features[4])
        self.assertIsNotNone(output,'_predict_point doesn\'t return anything')

    def test__predict_point_1(self):
        print 'in test 3'
        output = self.map._predict_point_1(self.features[12],10)
        self.assertIsNotNone(output,'_predict_point1 doesn\'t return anything')

    def test__get_BMU(self):
        print 'in test 4'
        output = self.map._get_BMU(self.features[12])
        self.assertIsNotNone(output,'_get_BMU doesn\'t return anything')
        print output.coords()

        min_distance = 9223372036854775807
        for neu in self.map.map_neurons.values():
            distance = np.linalg.norm(neu.weight_vs-self.features[12])          #np.linalg.norm was used because in _get_BMU minkowski distance was used with power of 2, which is the same as Euclidean distance
            if distance<min_distance:
                min_distance = distance
                coordinate = [str(neu.x_c),str(neu.y_c)]

        print np.array(coordinate)
        self.assertTrue(np.alltrue(output.coords()==np.array(coordinate)))


    def test__grow_map(self):
        print 'in test 5'

        for i in range(self.features.shape[0]):
            input = self.features[i]
            self.map._grow_map(input,i)

        self.assertTrue(len(self.map.map_neurons) > 4, 'map has not grown')

    def test__process_input(self):
        print 'in test 6'
        output = self.map._process_input(self.features[5])
        self.assertIsNotNone(output,'_process_input doesn\'t return anything')

        self.assertTrue(len(output) == 2)                       #check whether it returns coordinates of the BMU

    def test__type_b_weight_init(self):
        print 'in test 7'
        j = 0
        for neu in self.map.map_neurons.values():
            print neu.weight_vs;
            new_weight = self.map._type_b_weight_init(j,neu)
            j += 1
            print new_weight
            self.assertFalse(np.any(neu.weight_vs==new_weight),'the weight has not adjusted')
            print ' '

        print 'Number of neurons in the map',len(self.map.map_neurons)

    def test__type_b_weight_test(self):
        print 'in test 8'

        x=0
        y=2

        self.map.map_neurons[str(x)+""+str(y)] = Refactored_gsom.neuron(x,y,dims=16)
        self.map.map_neurons[str(x)+""+str(-y)] = Refactored_gsom.neuron(x,-y,dims=16)
        self.map.map_neurons[str(y)+""+str(x)] = Refactored_gsom.neuron(y,x,dims=16)
        self.map.map_neurons[str(-y)+""+str(x)] = Refactored_gsom.neuron(-y,x,dims=16)

        for neu in self.map.map_neurons.values():
            print neu.x_c, neu.y_c

        neuron = self.map.map_neurons[str(0)+""+str(0)]
        for i in range(0,4):
            output = self.map._type_b_weight_init(i,neuron)
            print output
            if i==0:
                new_weight = (self.map.map_neurons[str(x)+""+str(-y)].weight_vs + neuron.weight_vs)/2
                print new_weight
            elif i==1:
                new_weight = (self.map.map_neurons[str(x)+""+str(y)].weight_vs + neuron.weight_vs)/2
                print new_weight
            elif i==2:
                new_weight = (self.map.map_neurons[str(-y)+""+str(x)].weight_vs + neuron.weight_vs)/2
                print new_weight
            else:
                new_weight = (self.map.map_neurons[str(y)+""+str(x)].weight_vs + neuron.weight_vs)/2
                print new_weight

            self.assertTrue(np.alltrue(output==new_weight))

    def test__grow_type_b_weight_init(self):
        print 'in test 8'

        for i in range(self.features.shape[0]):                 #first grow the map and then adjust the weights and see every neurons' weight has adjusted
            input = self.features[i]
            self.map._grow_map(input,i)

        j = 0
        for neu in self.map.map_neurons.values():
            print neu.weight_vs;
            new_weight = self.map._type_b_weight_init(j,neu)
            j += 1
            print new_weight
            self.assertFalse(np.any(neu.weight_vs==new_weight),'the weight has not adjusted')
            print ' '

        print 'Number of neurons in the map',len(self.map.map_neurons)

    def test_type_a_weight_init(self):
        print 'in test 9'
        j = 0
        for neu in self.map.map_neurons.values():
            print neu.weight_vs;
            new_weight = self.map._type_a_weight_init(j,neu)
            j += 1
            print new_weight
            self.assertFalse(np.any(neu.weight_vs==new_weight),'the weight has not adjusted')
            print ' '

        print 'Number of neurons in the map',len(self.map.map_neurons)


    def test_grow_type_a_weight_init(self):
        print 'in test 10'

        for i in range(self.features.shape[0]):                 #first grow the map and then adjust the weights and see every neurons' weight has adjusted
            input = self.features[i]
            self.map._grow_map(input,i)

        j = 0
        for neu in self.map.map_neurons.values():
            print neu.weight_vs;
            new_weight = self.map._type_a_weight_init(j,neu)
            j += 1
            print new_weight
            self.assertFalse(np.any(neu.weight_vs==new_weight),'the weight has not adjusted')
            print ' '

        print 'Number of neurons in the map',len(self.map.map_neurons)

    def test_type_c_weight_init(self):
        print 'in test 11'
        j = 0
        for neu in self.map.map_neurons.values():
            print neu.weight_vs;
            new_weight = self.map._type_c_weight_init(j,neu)
            j += 1
            print new_weight
            self.assertFalse(np.any(neu.weight_vs==new_weight),'the weight has not adjusted')
            print ' '

        print 'Number of neurons in the map',len(self.map.map_neurons)


    def test_grow_type_c_weight_init(self):
        print 'in test 12'

        for i in range(self.features.shape[0]):                 #first grow the map and then adjust the weights and see every neurons' weight has adjusted
            input = self.features[i]
            self.map._grow_map(input,i)

        j = 0
        for neu in self.map.map_neurons.values():
            print neu.weight_vs;
            new_weight = self.map._type_c_weight_init(j,neu)
            j += 1
            print new_weight
            self.assertFalse(np.any(neu.weight_vs==new_weight),'the weight has not adjusted')
            print ' '

        print 'Number of neurons in the map',len(self.map.map_neurons)


if __name__ == '__main__':
    unittest.main()