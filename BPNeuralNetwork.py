import random
import math
# import numpy as np

class BPNeuralNetwork:
    def __init__(self, learnRate, momentum, layers, layerSpec):
        self.learnRate = learnRate
        self.momentum = momentum
        self.layers = layers
        self.layerSpec = layerSpec

        # Neuron outputs
        self.out = [None] * layers
        for i in range(layers):
            self.out[i] = [None] * layerSpec[i]

        # Differences
        self.delta = [None] * layers
        for i in range(layers):
            self.delta[i] = [None] * layerSpec[i]

        # Weights
        self.weight = [None] * layers
        for i in range(layers):
            self.weight[i] = [None] * layerSpec[i]
            for j in range(layerSpec[i]):
                if i > 0:
                    self.weight[i][j] = [None] * (layerSpec[i - 1] + 1)

        for i in range(layers):
            for j in range(layerSpec[i]):
                if i > 0:
                    for k in range(layerSpec[i - 1] + 1):
                        self.weight[i][j][k] = (-1.0 + (random.random() * 2.0))

        # WeightChange
        self.weightChange = [None] * layers
        for i in range(layers):
            self.weightChange[i] = [None] * layerSpec[i]
            for j in range(layerSpec[i]):
                if i > 0:
                    self.weightChange[i][j] = [None] * (layerSpec[i - 1] + 1)

        for i in range(layers):
            for j in range(layerSpec[i]):
                if i > 0:
                    for k in range(layerSpec[i - 1] + 1):
                        self.weightChange[i][j][k] = 0.0


    def transferFunction(self, input):
        return 1 / (1 + math.exp(-input))  # sigmoid

    def transferFunctionDerivative(self, input):
        return (1.0 - self.transferFunction(input)) * self.transferFunction(input)  # sigmoid derivative

    def feedForward(self, input):
        # Move input data to the first layer
        for i in range(self.layerSpec[0]):
            self.out[0][i] = input[i]

        # Calculate output for each layer
        for i in range(self.layers):
            for j in range(self.layerSpec[i]):
                sum = 0.0
                if i > 0:
                    for k in range(self.layerSpec[i - 1]):
                        sum += self.out[i - 1][k] * self.weight[i][j][k]
                    sum += self.weight[i][j][self.layerSpec[i - 1]]

                    self.out[i][j] = self.transferFunction(sum)

    def backPropagate(self, input, target):
        self.feedForward(input)

        # Calculate output differences
        for i in range(self.layerSpec[self.layers-1]):
            self.delta[self.layers - 1][i] = self.transferFunctionDerivative(self.out[self.layers - 1][i]) * (target[i] - self.out[self.layers - 1][i])

        # Calculate differences for previous layers
        for i in range(self.layers - 2, 0, -1):
            for j in range(self.layerSpec[i]):
                sum = 0.0
                for k in range(self.layerSpec[i + 1]):
                    sum += self.delta[i + 1][k] * self.weight[i + 1][k][j]
                self.delta[i][j] = self.transferFunctionDerivative(self.out[i][j]) * sum

        # Add some momentum... if any :) ...to keep it moving!
        for i in range(1, self.layers):
            for j in range(self.layerSpec[i]):
                for k in range(self.layerSpec[i - 1]):
                    self.weight[i][j][k] += self.momentum * self.weightChange[i][j][k]
                # For the BIAS too
                self.weight[i][j][self.layerSpec[i-1]] += self.momentum * self.weightChange[i][j][self.layerSpec[i-1]]

        # Update weights and weightChanges using the learnRate and the difference.
        for i in range(1, self.layers):
            for j in range(self.layerSpec[i]):
                for k in range(self.layerSpec[i-1]):
                    self.weightChange[i][j][k] = self.learnRate * self.delta[i][j] * self.out[i - 1][k]
                    self.weight[i][j][k] += self.weightChange[i][j][k]
                # Some love for the BIAS too...
                self.weightChange[i][j][self.layerSpec[i-1]] = self.learnRate * self.delta[i][j]
                self.weight[i][j][self.layerSpec[i - 1]] += self.weightChange[i][j][self.layerSpec[i-1]]

    def outValue(self, position):
        return self.out[self.layers - 1][position]

    def meanSquareError(self, target):
        mse = 0.0

        # The SUM
        for i in range(self.layerSpec[self.layers - 1]):
            mse += (target[i] - self.out[self.layers - 1][i]) ** 2

        # The 1/n
        mse = mse / self.layerSpec[self.layers - 1]

        return mse

    def getNetWeights(self):
        weightString = ""

        for i in range(1, self.layers):
            for j in range(self.layerSpec[i]):
                for k in range(self.layerSpec[i - 1] + 1):
                    weightString += str(self.weight[i][j][k]) + ";"

        return weightString

    def setNetWeights(self, weightString):
        weightStrings = weightString.split(";")
        nextStringValue = 0

        for i in range(1, self.layers):
            for j in range(self.layerSpec[i]):
                for k in range(self.layerSpec[i - 1] + 1):
                    self.weight[i][j][k] = float(weightStrings[nextStringValue])
                    nextStringValue += 1

def teach_and_test():
    # Creates a new network
    bp_network = BPNeuralNetwork(0.1, 0.5, 4, [4, 64, 64, 16])

    # Input data
    input_data = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]

    # The output data contains the requested result from the network
    out_data = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

    max_iterations = 2000000
    iterations = 0
    b_trained = False
    train_cycles = 0
    comulated_distance = 0.0
    thresh = 0.001

    print("Training the NeuralNetwork...")

    for iterations in range(max_iterations):
        if b_trained:
            break

        b_trained = True
        for i_buff_learn in range(16):
            for i_inner_learn_cycle in range(10):
                # Do a back propagation process (learning) for this sample
                bp_network.backPropagate(
                    input_data[i_buff_learn],
                    out_data[i_buff_learn]
                )

                # Get the error rate for this sample
                d_temp = bp_network.meanSquareError(out_data[i_buff_learn])

                comulated_distance += d_temp
                train_cycles += 1

                if d_temp <= (comulated_distance / train_cycles) or d_temp < thresh:
                    break

            if d_temp >= thresh:
                # If any training sample fails - let's continue the learning
                b_trained = False

        if iterations % (max_iterations / 100) == 0:
            print(
                f"Still training... last meanSquareError: {d_temp}; average meanSquareError: {(comulated_distance / train_cycles)}")

    print(
        f"{train_cycles} trainCycle completed... in {iterations} iteration cycles... average meanSquareError: {(comulated_distance / train_cycles)}")

    weight_string = bp_network.getNetWeights()
    print("begin weightString")
    print(weight_string)
    print("end weightString")

    bp_network = BPNeuralNetwork(0.1, 0.5, 4, [4, 64, 64, 16])
    bp_network.setNetWeights(weight_string)

    print("Testing the network...")

    for i_buff_learn in range(16):
        bp_network.feedForward(input_data[i_buff_learn])

        result_message = f"{i_buff_learn} "

        for i_out in range(16):
            d_out = bp_network.outValue(i_out)

            if d_out >= 1.1:
                result_message += "H "
            elif d_out >= 0.8:
                result_message += "1 "
            elif d_out <= 0.2:
                result_message += "0 "
            elif d_out <= 0:
                result_message += "L "
            else:
                result_message += "M "

        print(result_message)

def test_only():
    bp_network = BPNeuralNetwork(0.1, 0.5, 4, [4, 8, 8, 16])

    weight_string = "4.339822215767142;2.545060808733524;1.207030630675552;3.366778609372262;-2.464595048011004;-3.3381035015654463;4.11716774696896;0.6530644413652703;-4.212873421401852;-0.10975277271373848;4.9325405029051925;-0.04169158934020014;0.22973637782821757;-4.373964084775681;-0.1702072947751098;3.85337288189308;6.345330110338298;5.198269841202475;-2.3728199823599745;-3.4863554518218325;1.4547494190454404;1.6351174415108258;-0.19639588802094765;1.113511433760181;3.5999524377765697;2.8857116495813657;1.0618052632230373;3.911426555187221;2.7103437920014413;-9.240771944754115;0.33168652907358825;3.5795003698037196;0.19273501192320902;4.3845901663105895;-2.100976269199784;-3.0067069931045887;4.4681582611176385;-2.962665898616419;0.8215558118412006;-1.2459503297203296;1.746462421997402;1.4003221934288987;1.8481725276877932;1.751487473546771;1.829170138108642;1.2706928412295577;1.8071742260344974;0.703963183947759;1.936450430263346;0.914941514055242;1.6162935483533822;1.5511915701563779;1.589110694170343;1.6820417961943885;1.2251703320880907;2.2050615305190457;1.3043581845373116;1.8029947450221824;-0.9220097480305276;3.2330211600081302;-7.041756946565191;8.92252906499992;-3.3361317522593974;-1.1448649798925299;3.480501082351243;7.606754716335958;-3.158932033301973;5.313083170978849;-4.018261801605604;0.8977024420970947;2.014270178484873;-3.2436310781968145;8.89801982304988;3.852654678099841;-1.0284322710099476;-5.216322699814397;-0.7546785531809852;-0.08004383618716092;12.28916493740847;4.805701844010844;-2.5168353587519467;-6.632623633393037;-2.841043794400226;-3.7831875060584395;-2.0303335858508635;2.473075885985218;1.476761978721395;1.2715566767481579;2.1675395213126194;0.9643928093159285;1.0223257618719834;1.7066341047803628;0.6616815277148065;1.6011659837589494;2.066248413426922;2.243355343438376;1.656861839914691;1.700879280213542;2.2294574952216144;0.5308978085072323;0.6154145466838002;0.35718575629856747;1.6470145042123987;1.6916050367263524;1.1844401907198239;0.761032467673819;1.6145313606798062;2.6699891862739267;1.3791270068206358;1.3444147028939148;0.5316493567250592;1.4793960641346258;1.2117507841941335;0.8137677774631454;-11.520042828185295;-28.856544262320536;-2.7853418942296635;-0.3765778540241101;1.4224430427523773;-0.16015155628732125;1.7105167693056986;0.6764512888087963;1.933442124418097;-7.854960249145277;-11.650584447553088;-21.003463414784708;1.9998454088924793;0.40129411159838646;0.6078908217929717;1.6141386656287695;-2.0390534397598783;-2.5621660691820813;-0.9898049780055912;-28.430579480668523;15.621581227336986;-1.4456231606427241;-2.243200909588349;-1.6154578359815168;-2.6189901622117264;-0.5089754229094942;0.125210037385578;-9.217406841233965;10.562050309496668;-21.737468660530354;-0.6538707545148666;-0.9951421043645706;-0.8578062363218522;0.3849523130566711;-0.32488563503136153;0.14737556459307236;6.376581299086911;-28.94090763656841;-8.702322355422174;0.5536172028389451;0.2554657380674933;-0.3992625704955787;-1.5128146616385254;-1.4982878433776077;-0.4847658070502034;9.832891855608136;-11.293551994524266;-28.606751511449758;-0.8168316051453445;-0.45602849075919116;-0.37069099444767867;-0.7618822427162587;-3.98905802509239;-3.287927905428314;18.167408624791907;-19.03771626392233;12.346575587211428;-3.9673756393166006;-4.105374389181567;-3.9602229954256987;-5.051820440398523;-2.3231012335233956;-2.8372095961282784;10.55390277970135;12.845833575367202;-25.813419279322474;-2.780753705247779;-2.060117745397721;-3.4086969816765076;-3.730435616048142;-2.4796326557840627;-1.610872983396143;-17.78312732647966;-12.817816357640746;18.536636493251727;-2.7845289034747442;-1.7323097844118218;-2.984632335699555;-1.4792712931709693;-2.225149294677998;-1.5227194268658197;-24.06056409774177;16.186155235086037;-5.713692348640345;-0.7756817566594162;-2.038067967465165;-0.5529921578176873;-0.839550905400449;-3.2093667535921715;-3.467897877242098;-16.192904363272223;12.462679760217318;16.26727036921776;-2.7166054772316715;-4.4650675478616115;-3.821048673881837;-4.1343845162166515;-5.907643641543409;-4.415226486836968;-8.157786422490563;31.877528356240926;6.322137843226262;-5.855812868701539;-4.384161574882401;-6.123616297829962;-4.589929221762815;-6.327794165922985;-6.464483989588365;15.189555139825325;0.9605145231841425;22.61615144348559;-6.352531693287689;-5.392687022407387;-5.571891647102274;-5.2687404142351575;-5.3631297784198715;-4.62691960731446;15.5763228683039;15.106160811970708;4.682121618039366;-5.439926541739246;-5.54000959503483;-3.7397410180740382;-5.431412585902381;-6.678584670298125;-8.08839226042406;3.9240522410175593;18.11818789142456;25.612331872674208;-6.389715458903294;-6.335248145434048;-6.392703523742943;-6.86827397297405;-4.042650633985365;-5.117048621599839;0.8171562647390781;31.922746178641173;-11.581483788347608;-4.695819522718994;-5.066557056052905;-3.494253209435038;-4.828395133263423;"
    bp_network.setNetWeights(weight_string)

    input_data = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]

    print("Testing the network...")

    for i_buff_learn in range(16):
        bp_network.feedForward(input_data[i_buff_learn])
        result_message = f"{i_buff_learn} "

        for i_out in range(16):
            d_out = bp_network.outValue(i_out)

            if d_out >= 1.1:
                result_message += "H "
            elif d_out >= 0.8:
                result_message += "1 "
            elif d_out <= 0.2:
                result_message += "0 "
            elif d_out <= 0:
                result_message += "L "
            else:
                result_message += "M "

        print(result_message)


teach_and_test()
test_only()
