package org.ml4j.nn.components.builders.common;

import java.util.function.Supplier;

public class PathEnderImpl<P, C> implements PathEnder<P, C> {

	protected P previous;
	private Supplier<C> newPathCreator;
	
	public PathEnderImpl(P previous, Supplier<C> newPathCreator) {
		this.previous = previous;
		this.newPathCreator = newPathCreator;
	}
	
	@Override
	public P endParallelPaths() {
		//System.out.println("Ending parallel paths:" + this.getClass().getName());
		onParallelPathsEnd();
		return previous;
	}
	
	protected void onParallelPathsEnd() {
		
	}

	@Override
	public C withPath() {
		//System.out.println("Created new path");
		throw new UnsupportedOperationException("Multiple paths not yet supported");
		//onPathEnd();
		//return newPathCreator.get();
	}

	protected void onPathEnd() {
		// TODO Auto-generated method stub
		
	}

}
