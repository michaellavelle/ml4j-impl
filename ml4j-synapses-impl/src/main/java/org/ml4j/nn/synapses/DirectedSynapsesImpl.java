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

package org.ml4j.nn.synapses;

import java.util.Arrays;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.DirectedAxonsComponent;
import org.ml4j.nn.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.axons.DirectedAxonsComponentImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedDipoleGraphImpl;
import org.ml4j.nn.graph.DirectedPath;
import org.ml4j.nn.graph.DirectedPathImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapses.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesImpl<L extends Neurons, R extends Neurons> 
    implements DirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedSynapsesImpl.class);
  
  private Axons<? extends L, ? extends R, ?> primaryAxons;
  private DifferentiableActivationFunction activationFunction;
  private DirectedDipoleGraph<DirectedAxonsComponent<?, ?>> axonsGraph;
  
  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param primaryAxons The primary Axons within these synapses
   * @param axonsGraph The axons graph within these Synapses.
   * @param activationFunction The activation function within these synapses
   */
  protected DirectedSynapsesImpl(Axons<? extends L, ? extends R, ?> primaryAxons, 
      DirectedDipoleGraph<DirectedAxonsComponent<?, ?>> axonsGraph, 
      DifferentiableActivationFunction activationFunction) {
    super();
    this.primaryAxons = primaryAxons;
    this.activationFunction = activationFunction;
    this.axonsGraph = axonsGraph;
  }
  
  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param primaryAxons The primary Axons within these synapses
   * @param activationFunction The activation function within these synapses
   */
  public DirectedSynapsesImpl(Axons<? extends L, ? extends R, ?> primaryAxons, 
      DifferentiableActivationFunction activationFunction) {
      this(primaryAxons, new DirectedDipoleGraphImpl<DirectedAxonsComponent<? ,?>>(new DirectedAxonsComponentImpl<>(primaryAxons)), 
          activationFunction);
  }
  

  @Override
  public Axons<? extends L, ? extends R, ?> getPrimaryAxons() {
    return primaryAxons;
  }
  
  /**
   * @return The Axons graph within these DirectedSynapses.
   */
  public DirectedDipoleGraph<DirectedAxonsComponent<?, ?>> getAxonsGraph() {
    return axonsGraph;
  }

  @Override
  public DirectedSynapses<L, R> dup() {
   return new DirectedSynapsesImpl<>(primaryAxons.dup(), cloneAxonsGraph() , 
        activationFunction);
  }
  
  private DirectedDipoleGraph<DirectedAxonsComponent<?, ?>> cloneAxonsGraph() {

    DirectedDipoleGraph<DirectedAxonsComponent<?, ?>> dup = new DirectedDipoleGraphImpl<>();
    for (DirectedPath<DirectedAxonsComponent<?, ?>> directedPath : axonsGraph.getParallelPaths()) {
      DirectedPath<DirectedAxonsComponent<?, ?>> dupPath = new DirectedPathImpl<>();
      for (DirectedAxonsComponent<?, ?> axonsComponent : directedPath.getEdges()) {
        Axons<?, ?, ?> dupAxons = axonsComponent.getAxons().dup();
        dupPath.addEdge(new DirectedAxonsComponentImpl<Neurons, Neurons>(dupAxons));
      }
      dup.addParallelPath(dupPath);
    }
    return dup;
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return activationFunction;
  }


  @Override
  public DirectedSynapsesActivation forwardPropagate(NeuronsActivation input,
      DirectedComponentsContext synapsesContext) {

    LOGGER.debug("Forward propagating through DirectedSynapses");
    
    NeuronsActivation inputNeuronsActivation = input;

    Matrix totalAxonsOutputMatrix = null;
    
    NeuronsActivation axonsOutputActivation = null;

    DirectedDipoleGraph<DirectedAxonsComponentActivation> axonsActivationGraph =
        new DirectedDipoleGraphImpl<DirectedAxonsComponentActivation>();

    int pathIndex = 0;
    
    for (DirectedPath<DirectedAxonsComponent<?, ?>> parallelAxonsPath : this.getAxonsGraph().getParallelPaths()) {

      DirectedPath<DirectedAxonsComponentActivation> axonsActivationPath = new DirectedPathImpl<DirectedAxonsComponentActivation>();

      int axonsIndex = 0;
      
      for (DirectedAxonsComponent<?, ?> axonsComponent : parallelAxonsPath.getEdges()) {
    	  
      DirectedAxonsComponentActivation axonsComponentActivation = axonsComponent.forwardPropagate(inputNeuronsActivation, axonsComponent.getContext(synapsesContext, axonsIndex));

       // AxonsActivation axonsActivation =
         //   axons.pushLeftToRight(inputNeuronsActivation, null, 
           // 		synapsesContext.getAxonsContext(pathIndex, axonsIndex));

        axonsActivationPath.addEdge(axonsComponentActivation);
        axonsOutputActivation = axonsComponentActivation.getOutput();
        inputNeuronsActivation = axonsOutputActivation;
        axonsIndex++;
      }
      if (totalAxonsOutputMatrix == null) {
        totalAxonsOutputMatrix = inputNeuronsActivation.getActivations();
      } else {
        
        Matrix axonsPathOutputActivationMatrix = axonsOutputActivation.getActivations();
        
        if (axonsPathOutputActivationMatrix.getRows() != totalAxonsOutputMatrix.getRows()) {
          throw new IllegalStateException(
              "Final axons activation in each parallel path must be the "
              + "same dimensions");
        }
        if (axonsPathOutputActivationMatrix.getColumns() 
            != totalAxonsOutputMatrix.getColumns()) {
          throw new IllegalStateException(
              "Final axons activation in each parallel path must be the " + "same dimensions");
        }
        
        totalAxonsOutputMatrix =
            totalAxonsOutputMatrix.add(axonsPathOutputActivationMatrix);
      }
      axonsActivationGraph.addParallelPath(axonsActivationPath);
      pathIndex++;
    }
    
    NeuronsActivation totalAxonsOutputActivation = new NeuronsActivation(totalAxonsOutputMatrix, 
        axonsOutputActivation.getFeatureOrientation());
    
    DifferentiableActivationFunctionComponent actComp = new DifferentiableActivationFunctionDirectedComponentImpl(activationFunction);
    
    //DifferentiableActivationFunctionActivation activationFunctionActivation =
      //  activationFunction.activate(totalAxonsOutputActivation, synapsesContext);
    DifferentiableActivationFunctionActivation actAct = actComp.forwardPropagate(totalAxonsOutputActivation, new NeuronsActivationContext() {

		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		@Override
		public MatrixFactory getMatrixFactory() {
			return synapsesContext.getMatrixFactory();
		}
    	
    });

    NeuronsActivation outputNeuronsActivation = actAct.getOutput();

    return new DirectedSynapsesActivationImpl(this, input, axonsActivationGraph,
    		actAct, outputNeuronsActivation, synapsesContext);

  }

  @Override
  public L getLeftNeurons() {
    return primaryAxons.getLeftNeurons();
  }

  @Override
  public R getRightNeurons() {
    return primaryAxons.getRightNeurons();
  }

@Override
public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
	return directedComponentsContext; 
}

@Override
public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
	return Arrays.asList(new DirectedAxonsComponentImpl<>(primaryAxons), new DifferentiableActivationFunctionDirectedComponentImpl (this.activationFunction));
}

}
