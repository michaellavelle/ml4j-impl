/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.ScaleAndShiftAxonsConfig;
import org.ml4j.nn.axons.ScaleAndShiftAxonsImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.synapses.BatchNormDirectedSynapses;
import org.ml4j.nn.synapses.BatchNormDirectedSynapsesImpl;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesChain;
import org.ml4j.nn.synapses.DirectedSynapsesChainImpl;
import org.ml4j.nn.synapses.DirectedSynapsesImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A default base implementation of FeedForwardLayer.
 * 
 * @author Michael Lavelle
 * 
 * @param <A> The type of primary Axons in this FeedForwardLayer.
 */
public abstract class FeedForwardLayerBase<A extends Axons<?, ?, ?>,
    L extends FeedForwardLayer<A, L>> extends AbstractFeedForwardLayer<A, L>
    implements FeedForwardLayer<A, L> {

/**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(FeedForwardLayerBase.class);

  protected A primaryAxons;
  
  protected DifferentiableActivationFunction primaryActivationFunction;
    
  protected boolean withBatchNorm;
   
  /**
   * @param primaryAxons The primary Axons
   * @param activationFunction The primary activation function
   * @param matrixFactory The matrix factory
   * @param withBatchNorm Whether to enable batch norm.
   */
  protected FeedForwardLayerBase(A primaryAxons, 
      DifferentiableActivationFunction activationFunction, MatrixFactory matrixFactory, 
      boolean withBatchNorm) {
	  super(new DirectedSynapsesChainImpl<>(getSynapses(matrixFactory, primaryAxons, activationFunction, withBatchNorm)), matrixFactory);
    this.primaryAxons = primaryAxons;
    this.primaryActivationFunction = activationFunction;
    this.matrixFactory = matrixFactory;
    this.withBatchNorm = withBatchNorm;
  }

  @Override
  public int getInputNeuronCount() {
    return primaryAxons.getLeftNeurons().getNeuronCountIncludingBias();
  }

  @Override
  public int getOutputNeuronCount() {
    return primaryAxons.getRightNeurons().getNeuronCountIncludingBias();
  }

  @Override
  public A getPrimaryAxons() {
    return primaryAxons;
  }
  
  @Override
  public DifferentiableActivationFunction getPrimaryActivationFunction() {
    return primaryActivationFunction;
  }
  
  protected static List<DirectedSynapses<?, ?>> getSynapses(MatrixFactory matrixFactory, Axons<?, ?, ?> primaryAxons, DifferentiableActivationFunction primaryActivationFunction, boolean withBatchNorm) {
	    List<DirectedSynapses<?, ?>> synapses = new ArrayList<>();
	    if (withBatchNorm) {
	    	      
	    	  Matrix initialGamma = matrixFactory.createOnes(1, 
	    			  primaryAxons.getRightNeurons().getNeuronCountExcludingBias());
	    	  Matrix initialBeta = matrixFactory.createZeros(1, 
	    			  primaryAxons.getRightNeurons().getNeuronCountExcludingBias());
	    	  ScaleAndShiftAxonsConfig config = 
	    	          new ScaleAndShiftAxonsConfig(initialGamma, initialBeta);
	    	      
	    	  BatchNormDirectedSynapses<?, ?> batchNormSynapses = new BatchNormDirectedSynapsesImpl<Neurons, Neurons>(
	    	    		  primaryAxons.getRightNeurons(), primaryAxons.getRightNeurons(),
	    	          new ScaleAndShiftAxonsImpl(
	    	              new Neurons(primaryAxons.getRightNeurons().getNeuronCountExcludingBias(), true),
	    	              primaryAxons.getRightNeurons(), matrixFactory, config), primaryActivationFunction);
	    	
	    	
	      synapses.add(new DirectedSynapsesImpl<Neurons, Neurons>(primaryAxons, 
	          new LinearActivationFunction()));
	      
	      synapses.add(batchNormSynapses);
	      
	    } else {
	      synapses.add(new DirectedSynapsesImpl<Neurons, Neurons>(
	          primaryAxons, primaryActivationFunction));
	    }
	    return synapses;
	  }
	  
	  @Override
	  public List<ChainableDirectedComponent<NeuronsActivation, ?, ?>> getComponents() {
		  List<ChainableDirectedComponent<NeuronsActivation, ?, ?>> components = new ArrayList<>();
		  components.addAll(getSynapses(matrixFactory, primaryAxons, primaryActivationFunction, withBatchNorm));
		  return components;
	  }
   
	protected TrailingActivationFunctionDirectedComponentChain<?> createChain() {
		DirectedSynapsesChain<DirectedSynapses<?, ?>> synapseChain = new DirectedSynapsesChainImpl<>(getSynapses(matrixFactory, primaryAxons, primaryActivationFunction, withBatchNorm));
		List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> chainableComponents = new ArrayList<>();
		chainableComponents.addAll(synapseChain.decompose());
		return new TrailingActivationFunctionDirectedComponentChainImpl(chainableComponents);
	}
	
  
  @Override
  public DirectedLayerActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      DirectedLayerContext directedLayerContext) {
    LOGGER.debug(directedLayerContext.toString() + ":Forward propagating through layer");
    
	DirectedComponentsContext componentsContext = new DirectedComponentsContextImpl(directedLayerContext.getMatrixFactory()); 

    TrailingActivationFunctionDirectedComponentChainActivation activation = createChain().forwardPropagate(inputNeuronsActivation, componentsContext);
    
	return new DirectedLayerActivationImpl(this, activation, directedLayerContext);   
  }
  
  @Override
  public DirectedLayerContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
	  return directedComponentsContext.getContext(this, () -> new DirectedLayerContextImpl(componentIndex, matrixFactory));
  }
  

  @Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
	return getComponents().stream().flatMap(c -> c.decompose().stream()).collect((Collectors.toList()));
  }

@Override
  public NeuronsActivation getOptimalInputForOutputNeuron(int outputNeuronIndex,
      DirectedLayerContext directedLayerContext) {
    LOGGER.debug("Obtaining optimal input for output neuron with index:" + outputNeuronIndex);
    Matrix weights = getPrimaryAxons().getDetachedConnectionWeights();
    int countJ = weights.getRows() - (getPrimaryAxons().getLeftNeurons().hasBiasUnit() ? 1 : 0);
    double[] maximisingInputFeatures = new double[countJ];
    boolean hasBiasUnit = getPrimaryAxons().getLeftNeurons().hasBiasUnit();

    for (int j = 0; j < countJ; j++) {
      double wij = getWij(j, outputNeuronIndex, weights, hasBiasUnit);
      double sum = 0;

      if (wij != 0) {

        for (int j2 = 0; j2 < countJ; j2++) {
          double weight = getWij(j2, outputNeuronIndex, weights, hasBiasUnit);
          if (weight != 0) {
            sum = sum + Math.pow(weight, 2);
          }
        }
        sum = Math.sqrt(sum);
      }
      maximisingInputFeatures[j] = wij / sum;
    }
    return new NeuronsActivation(
        directedLayerContext.getMatrixFactory()
            .createMatrix(new double[][] {maximisingInputFeatures}),
         NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }
  
  private double getWij(int indI, int indJ, Matrix weights, boolean hasBiasUnit) {
    int indICorrected = indI + (hasBiasUnit ? 1 : 0);
    return weights.get(indICorrected, indJ);
  }
}
