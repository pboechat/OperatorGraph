


#ifndef INCLUDED_TECHNIQUE_INTERFACE_H
#define INCLUDED_TECHNIQUE_INTERFACE_H

#pragma once

#include <memory>
#include <string>

#include "types.h"

#include "interface.h"

#include "types.h"



class INTERFACE TechniqueInterface
{
protected:
	TechniqueInterface() {}
	TechniqueInterface(const TechniqueInterface&) {}
	TechniqueInterface& operator =(const TechniqueInterface&) { return *this; }
	~TechniqueInterface() {}
public:

	virtual void init() = 0;
	virtual void resetQueue() = 0;
	virtual void recordQueue() = 0;
	virtual void restoreQueue() = 0;
	virtual void insertIntoQueue(int num) = 0;
	virtual double execute(int phase = 0, double timelimit = 0) = 0;
	virtual std::string name() const = 0;
	virtual void release() = 0; 

	virtual int BlockSize(int phase = 0) const = 0;
	virtual int Blocks(int phase = 0) const = 0;
	virtual uint SharedMem(int phase = 0) const = 0;
};


template<class Technique, class FillFunc>
class TechniqueWrapper : public TechniqueInterface
{
protected:
  Technique technique;

  TechniqueWrapper(const TechniqueWrapper&) {}
  TechniqueWrapper& operator =(const TechniqueWrapper&) { return *this; }

public:

  TechniqueWrapper() {}
  ~TechniqueWrapper() {}

  void init() { technique.init(); }
  void resetQueue() { technique.resetQueue(); }
  void recordQueue() { technique.recordQueue(); }
  void restoreQueue() { technique.restoreQueue(); }
  void insertIntoQueue(int num) { technique. template insertIntoQueue<FillFunc>(num); }
  double execute(int phase = 0, double timelimit = 0) { return technique.execute(phase, timelimit); }
  std::string name() const { return technique.name(); }
	void release() { delete this; }

  int BlockSize(int phase = 0) const { return technique.BlockSize(phase); }
  int Blocks(int phase = 0) const { return technique.Blocks(phase); }
  uint SharedMem(int phase = 0) const { return technique.SharedMem(phase); }
};

template<template<int /*Phase*/> class LaunchTimelimitMicorSecondsTraits, class Technique, class FillFunc>
class TechniqueTimedWrapper : public TechniqueInterface
{
protected:
  Technique technique;
  TechniqueTimedWrapper(const TechniqueTimedWrapper&) {}
  TechniqueTimedWrapper& operator =(const TechniqueTimedWrapper&) { return *this; }
  
public:
  TechniqueTimedWrapper() {}
  ~TechniqueTimedWrapper() {}

  void init() { technique.init(); }
  void resetQueue() { technique.resetQueue(); }
  void recordQueue() { technique.recordQueue(); }
  void restoreQueue() { technique.restoreQueue(); }
  void insertIntoQueue(int num) { technique. template insertIntoQueue<FillFunc>(num); }
  double execute(int phase = 0) { return technique.execute(phase); }
  double execute(int phase, double timelimit) { return execute(phase); }
  std::string name() const { return technique.name(); }
	void release() { delete this; }

  int BlockSize(int phase = 0) const { return technique.BlockSize(phase); }
  int Blocks(int phase = 0) const { return technique.Blocks(phase); }
  uint SharedMem(int phase = 0) const { return technique.SharedMem(phase); }
};

struct technique_deleter
{
	template <typename Technique>
	void operator()(Technique* t)
	{
		if (t)
			t->release();
	}
};

struct technique_interface_deleter
{
	void operator()(TechniqueInterface* t)
	{
		if (t)
			t->release();
	}
};

typedef std::unique_ptr<TechniqueInterface, technique_interface_deleter> technique_ptr;

template <class Technique>
struct TechniqueCreator
{
	static technique_ptr create()
	{
		return technique_ptr(new Technique());
	}
};




#endif  // INCLUDED_TECHNIQUE_INTERFACE
