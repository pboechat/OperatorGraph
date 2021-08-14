#pragma once

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace PGA
{
	template <typename T, unsigned int N>
	class BaseDoublyLinkedList
	{
	protected:
		struct Node
		{
			int previous;
			int next;
			T value;

			__host__ __device__ Node() : previous(-2), next(-2) {}

		};

		unsigned int _size;
		Node nodes[N];

		__host__ __device__ __inline__ int newNode()
		{
			if (_size == N)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("BaseDoublyLinkedList::newNode(): list depleted");
				asm("trap;");
			}
#else
				throw std::runtime_error("BaseDoublyLinkedList::newNode(): list depleted");
#endif
			for (int i = 0; i < N; i++)
				if (nodes[i].previous == -2 && nodes[i].next == -2)
					return i;
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (_size == N)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("BaseDoublyLinkedList::newNode(): list is not depleted but couldn't find vacant node");
				asm("trap;");
			}
#else
				throw std::runtime_error("BaseDoublyLinkedList::newNode(): list is not depleted but couldn't find vacant node");
#endif
#endif
			return -1;
		}

		__host__ __device__ BaseDoublyLinkedList() : _size(0) {}

	public:
		__host__ __device__ __inline__ unsigned int size() const
		{
			return _size;
		}

		__host__ __device__ __inline__ T operator[](unsigned int i) const
		{
			auto& n = nodes[i];
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (n.next == -2 || n.previous == -2)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("BaseDoublyLinkedList::operator[]: accessing deleted or uninitialized node");
				asm("trap;");
			}
#else
				throw std::runtime_error("BaseDoublyLinkedList::operator[]: accessing deleted or uninitialized node");
#endif
#endif
			return n.value;
		}

		__host__ __device__ __inline__ int previous(unsigned int i) const
		{
			auto& n = nodes[i];
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (n.next == -2 || n.previous == -2)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("BaseDoublyLinkedList::previous(): accessing deleted or uninitialized node");
				asm("trap;");
			}
#else
				throw std::runtime_error("BaseDoublyLinkedList::previous(): accessing deleted or uninitialized node");
#endif
#endif
			return n.previous;
		}

		__host__ __device__ __inline__ int next(unsigned int i) const
		{
			auto& n = nodes[i];
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (n.next == -2 || n.previous == -2)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("BaseDoublyLinkedList::next(): accessing deleted or uninitialized node");
				asm("trap;");
			}
#else
				throw std::runtime_error("BaseDoublyLinkedList::next(): accessing deleted or uninitialized node");
#endif
#endif
			return n.next;
		}

		__host__ __device__ bool empty() const
		{
			return _size == 0;
		}

	};

	template <typename T, unsigned int N>
	class CircularDoublyLinkedList : public BaseDoublyLinkedList<T, N>
	{
	private:
		int ptr;

		__host__ __device__ __inline__ void addAfterNode(int i, int j, BaseDoublyLinkedList<T, N>::Node& n)
		{
			auto& ni = BaseDoublyLinkedList<T, N>::nodes[i];
			n.next = ni.next;
			n.previous = i;
			BaseDoublyLinkedList<T, N>::nodes[ni.next].previous = j;
			ni.next = j;
		}

	public:
		__host__ __device__ CircularDoublyLinkedList() : BaseDoublyLinkedList<T, N>(), ptr(-1) {}

		__host__ __device__ __inline__ int forwardSearch(T val)
		{
			int curr = ptr;
			unsigned int visited = 0;
			do
			{
				auto& n = BaseDoublyLinkedList<T, N>::nodes[curr];
				if (n.value == val)
					return curr;
				curr = n.next;
			} while (++visited < BaseDoublyLinkedList<T, N>::_size);
			return -1;
		}

		__host__ __device__ __inline__ int backwardSearch(T val)
		{
			int curr = ptr;
			unsigned int visited = 0;
			do
			{
				auto& n = BaseDoublyLinkedList<T, N>::nodes[curr];
				if (n.value == val)
					return curr;
				curr = n.previous;
			} while (++visited < BaseDoublyLinkedList<T, N>::_size);
			return -1;
		}

		__host__ __device__ __inline__ void addBack(T val)
		{
			int i = BaseDoublyLinkedList<T, N>::newNode();
			auto& n = BaseDoublyLinkedList<T, N>::nodes[i];
			n.value = val;
			if (ptr == -1)
			{
				n.previous = i;
				n.next = i;
			}
			else
				addAfterNode(ptr, i, n);
			ptr = i;
			BaseDoublyLinkedList<T, N>::_size++;
		}

		__host__ __device__ __inline__ void addFront(T val)
		{
			int i = BaseDoublyLinkedList<T, N>::newNode();
			auto& n = BaseDoublyLinkedList<T, N>::nodes[i];
			n.value = val;
			addAfterNode(ptr, i, n);
			BaseDoublyLinkedList<T, N>::_size++;
		}

		__host__ __device__ __inline__ void addAfter(int i, T val)
		{
			int j = BaseDoublyLinkedList<T, N>::newNode();
			auto& n = BaseDoublyLinkedList<T, N>::nodes[j];
			n.value = val;
			addAfterNode(i, j, n);
			BaseDoublyLinkedList<T, N>::_size++;
		}

		__host__ __device__ __inline__ void addBefore(int i, T val)
		{
			addAfter(BaseDoublyLinkedList<T, N>::nodes[i].previous, val);
			BaseDoublyLinkedList<T, N>::_size++;
		}

		__host__ __device__ __inline__ void remove(int i)
		{
			auto& n = BaseDoublyLinkedList<T, N>::nodes[i];
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (BaseDoublyLinkedList<T, N>::_size == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("CircularDoublyLinkedList::remove(): _size == 0");
				asm("trap;");
			}
#else
				throw std::runtime_error("CircularDoublyLinkedList::remove(): _size == 0");
#endif
			if (n.previous == -2 && n.next == -2)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("CircularDoublyLinkedList::remove(): invalid index");
				asm("trap;");
			}
#else
				throw std::runtime_error("CircularDoublyLinkedList::remove(): invalid index");
#endif
#endif
			if (n.next == i)
				ptr = -1;
			else
			{
				BaseDoublyLinkedList<T, N>::nodes[n.next].previous = n.previous;
				BaseDoublyLinkedList<T, N>::nodes[n.previous].next = n.next;
				if (i == ptr)
					ptr = n.previous;
			}
			n.next = -2;
			n.previous = -2;
			BaseDoublyLinkedList<T, N>::_size--;
		}

	};

	template <typename T, unsigned int N>
	class DoublyLinkedList : public BaseDoublyLinkedList<T, N>
	{
	private:
		int head, tail;

	public:
		__host__ __device__ DoublyLinkedList() : BaseDoublyLinkedList<T, N>(), head(-1), tail(-1) {}
		__host__ __device__ __inline__ int back() const
		{
			return tail;
		}

		__host__ __device__ __inline__ int front() const
		{
			return head;
		}

		__host__ __device__ __inline__ int forwardSearch(T val)
		{
			int curr = head;
			while (curr != -1)
			{
				auto& n = BaseDoublyLinkedList<T, N>::nodes[curr];
				if (n.value == val)
					return curr;
				curr = n.next;
			}
			return -1;
		}

		__host__ __device__ __inline__ int backwardSearch(T val)
		{
			int curr = tail;
			while (curr != -1)
			{
				auto& n = BaseDoublyLinkedList<T, N>::nodes[curr];
				if (n.value == val)
					return curr;
				curr = n.previous;
			}
			return -1;
		}

		__host__ __device__ __inline__ void addBack(T val)
		{
			int i = BaseDoublyLinkedList<T, N>::newNode();
			if (tail == -1)
			{
				auto& n = BaseDoublyLinkedList<T, N>::nodes[i];
				n.previous = -1;
				n.next = -1;
				n.value = val;
				head = tail = i;
			}
			else
			{
				auto& n = BaseDoublyLinkedList<T, N>::nodes[i];
				n.previous = tail;
				n.next = -1;
				n.value = val;
				BaseDoublyLinkedList<T, N>::nodes[tail].next = i;
				tail = i;
			}
			BaseDoublyLinkedList<T, N>::_size++;
		}

		__host__ __device__ __inline__ void addFront(T val)
		{
			int i = BaseDoublyLinkedList<T, N>::newNode();
			auto& n = BaseDoublyLinkedList<T, N>::nodes[i];
			if (head == -1)
			{
				n.previous = -1;
				n.next = -1;
				n.value = val;
				head = tail = i;
			}
			else
			{
				n.previous = -1;
				n.next = head;
				n.value = val;
				BaseDoublyLinkedList<T, N>::nodes[head].previous = i;
				head = i;
			}
			BaseDoublyLinkedList<T, N>::_size++;
		}

		__host__ __device__ __inline__ void remove(int i)
		{
			auto& n = BaseDoublyLinkedList<T, N>::nodes[i];
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (BaseDoublyLinkedList<T, N>::_size == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("DoublyLinkedList::remove(): _size == 0");
				asm("trap;");
			}
#else
				throw std::runtime_error("DoublyLinkedList::remove(): _size == 0");
#endif
			if (n.previous == -2 && n.next == -2)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			{
				printf("DoublyLinkedList::remove(): invalid index");
				asm("trap;");
			}
#else
				throw std::runtime_error("DoublyLinkedList::remove(): invalid index");
#endif
#endif
			if (n.previous == -1)
				head = n.next;
			else
				BaseDoublyLinkedList<T, N>::nodes[n.previous].next = n.next;
			if (n.next == -1)
				tail = n.previous;
			else
				BaseDoublyLinkedList<T, N>::nodes[n.next].previous = n.previous;
			n.next = -2;
			n.previous = -2;
			BaseDoublyLinkedList<T, N>::_size--;
		}

	};

}
