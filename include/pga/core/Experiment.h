#pragma once

#include <vector>
#include <limits>
#include <string>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include <cupti.h>

#include "CUDAException.h"
#include "CUPTIException.h"

#define getDeviceAttribute(__var, __device, __attribute) \
	{ \
		size_t size = sizeof(__var); \
		PGA::CUPTI::checkError(cuptiDeviceGetAttribute(__device, __attribute, &size, &__var)); \
	}

#define getEventGroupAttribute(__var, __event_group, __attribute) \
	{ \
		size_t size = sizeof(__var); \
		PGA::CUPTI::checkError(cuptiEventGroupGetAttribute(__event_group, __attribute, &size, &__var));  \
	}

#define getEventDomainAttribute3(__var, __event_domain, __attribute) \
	{ \
		size_t size = sizeof(__var); \
		PGA::CUPTI::checkError(cuptiEventDomainGetAttribute(__event_domain, __attribute, &size, &__var)); \
	}

#define getEventDomainAttribute4(__var, __device, __event_domain, __attribute) \
	{ \
		size_t size = sizeof(__var); \
		PGA::CUPTI::checkError(cuptiDeviceGetEventDomainAttribute(__device, __event_domain, __attribute, &size, &__var));  \
	}

#define getEventAttribute(__var, __evt, __attribute) \
	{ \
		size_t size = sizeof(__var); \
		PGA::CUPTI::checkError(cuptiEventGetAttribute(__evt, __attribute, &size, &__var)); \
	}

#define getStringEventAttribute(__var, __evt, __attribute) \
	{ \
		std::vector<char> buffer(256); \
		while (true) \
		{ \
			size_t size = buffer.size(); \
			CUptiResult res = cuptiEventGetAttribute(__evt, __attribute, &size, &buffer[0]); \
			if (res != CUPTI_SUCCESS) \
			{ \
				if (res == CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT) \
				{ \
					buffer.resize(buffer.size() * 2); \
					continue; \
				} \
				else \
					throw PGA::CUPTI::Exception(res); \
			} \
			break; \
		} \
		__var = &buffer[0]; \
	}

#define getMetricAttribute(__var, __metric, __attribute) \
	{ \
		size_t size = sizeof(__var); \
		PGA::CUPTI::checkError(cuptiMetricGetAttribute(__metric, __attribute, &size, &__var)); \
	}

#define getStringMetricAttribute(__var, __metric, __attribute) \
	{ \
		std::vector<char> buffer(256); \
		while (true) \
		{ \
			size_t size = buffer.size(); \
			CUptiResult res = cuptiMetricGetAttribute(__metric, __attribute, &size, &buffer[0]); \
			if (res != CUPTI_SUCCESS) \
			{ \
				if (res == CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT) \
				{ \
					buffer.resize(buffer.size() * 2); \
					continue; \
				} \
				else \
					throw PGA::CUPTI::Exception(res); \
			} \
			break; \
		} \
		__var = &buffer[0]; \
	}

namespace PGA
{
	namespace Instrumentation
	{
		std::vector<CUpti_EventDomainID> enumEventDomains(CUdevice device)
		{
			uint32_t num_domains;
			getDeviceAttribute(num_domains, device, CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID);
			std::vector<CUpti_EventDomainID> domains(num_domains);
			if (num_domains)
			{
				size_t domain_buffer_size = sizeof(CUpti_EventDomainID) * domains.size();
				PGA::CUPTI::checkError(cuptiDeviceEnumEventDomains(device, &domain_buffer_size, &domains[0]));
				domains.resize(domain_buffer_size / sizeof(CUpti_EventDomainID));
			}
			return domains;
		}

		uint32_t getNumMetrics()
		{
			uint32_t num_metrics;
			PGA::CUPTI::checkError(cuptiGetNumMetrics(&num_metrics));
			return num_metrics;
		}

		uint32_t getNumMetrics(CUdevice device)
		{
			uint32_t num_metrics;
			PGA::CUPTI::checkError(cuptiDeviceGetNumMetrics(device, &num_metrics));
			return num_metrics;
		}

		std::vector<CUpti_MetricID> enumMetrics()
		{
			auto num_metrics = getNumMetrics();
			std::vector<CUpti_MetricID> metrics(num_metrics);
			if (num_metrics)
			{
				size_t metric_buffer_size = sizeof(CUpti_MetricID) * metrics.size();
				PGA::CUPTI::checkError(cuptiEnumMetrics(&metric_buffer_size, &metrics[0]));
				metrics.resize(metric_buffer_size / sizeof(CUpti_MetricID));
			}
			return metrics;
		}

		std::vector<CUpti_MetricID> enumMetrics(CUdevice device)
		{
			auto num_metrics = getNumMetrics(device);
			std::vector<CUpti_MetricID> metrics(num_metrics);
			if (num_metrics)
			{
				size_t metric_buffer_size = sizeof(CUpti_MetricID) * metrics.size();
				PGA::CUPTI::checkError(cuptiDeviceEnumMetrics(device, &metric_buffer_size, &metrics[0]));
				metrics.resize(metric_buffer_size / sizeof(CUpti_MetricID));
			}
			return metrics;
		}

		uint32_t getNumEventsInMetric(CUpti_MetricID metric)
		{
			uint32_t num_events;
			PGA::CUPTI::checkError(cuptiMetricGetNumEvents(metric, &num_events));
			return num_events;
		}

		std::vector<CUpti_EventID> enumEventsInMetric(CUpti_MetricID metric)
		{
			auto num_events = getNumEventsInMetric(metric);
			std::vector<CUpti_EventID> events(num_events);
			if (num_events)
			{
				size_t event_buffer_size = sizeof(CUpti_EventID) * events.size();
				PGA::CUPTI::checkError(cuptiMetricEnumEvents(metric, &event_buffer_size, &events[0]));
				events.resize(event_buffer_size / sizeof(CUpti_EventID));
			}
			return events;
		}

		uint32_t getNumEventsInDomain(CUpti_EventDomainID domain)
		{
			uint32_t num_events;
			PGA::CUPTI::checkError(cuptiEventDomainGetNumEvents(domain, &num_events));
			return num_events;
		}

		std::vector<CUpti_EventID> enumEventsInDomain(CUpti_EventDomainID domain)
		{
			auto num_events = getNumEventsInDomain(domain);
			std::vector<CUpti_EventID> events(num_events);
			if (num_events)
			{
				size_t event_buffer_size = sizeof(CUpti_EventID) * events.size();
				PGA::CUPTI::checkError(cuptiEventDomainEnumEvents(domain, &event_buffer_size, &events[0]));
				events.resize(event_buffer_size / sizeof(CUpti_EventID));
			}
			return events;
		}

		void checkCUError(CUresult result)
		{

		}

		//////////////////////////////////////////////////////////////////////////
		class Experiment
		{
		private:
			static const char* translateEventCategory(CUpti_EventCategory category)
			{
				switch (category)
				{
				case CUPTI_EVENT_CATEGORY_INSTRUCTION:
					return "instruction";

				case CUPTI_EVENT_CATEGORY_MEMORY:
					return "memory";

				case CUPTI_EVENT_CATEGORY_CACHE:
					return "cache";

				case CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER:
					return "profile trigger";
				}
				return "invalid CUpti_EventCategory";
			}

			static const char* translateEventCollectionMethod(CUpti_EventCollectionMethod method)
			{
				switch (method)
				{
				case CUPTI_EVENT_COLLECTION_METHOD_PM:
					return "global hardware counter";

				case CUPTI_EVENT_COLLECTION_METHOD_SM:
					return "per MP hardware counter";

				case CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED:
					return "software instrumentation";
				}
				return "invalid CUpti_EventCollectionMethod";
			}

			static const char* translateMetricCategory(CUpti_MetricCategory category)
			{
				switch (category)
				{
				case CUPTI_METRIC_CATEGORY_MEMORY:
					return "memory";

				case CUPTI_METRIC_CATEGORY_INSTRUCTION:
					return "instruction";

				case CUPTI_METRIC_CATEGORY_MULTIPROCESSOR:
					return "multiprocessor";

				case CUPTI_METRIC_CATEGORY_CACHE:
					return "cache";

				case CUPTI_METRIC_CATEGORY_TEXTURE:
					return "texture";
				}
				return "invalid CUpti_MetricCategory";
			}

			static const char* translateMetricValueKind(CUpti_MetricValueKind value_kind)
			{
				switch (value_kind)
				{
				case CUPTI_METRIC_VALUE_KIND_DOUBLE:
					return "double";

				case CUPTI_METRIC_VALUE_KIND_UINT64:
					return "uint64";

				case CUPTI_METRIC_VALUE_KIND_PERCENT:
					return "percent";

				case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
					return "throughput";

				case CUPTI_METRIC_VALUE_KIND_INT64:
					return "int64";

				case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
					return "utilization";
				}
				return "invalid CUpti_MetricValueKind";
			}

			static const char* translateMetricEvaluationMode(CUpti_MetricEvaluationMode mode)
			{
				switch (mode)
				{
				case CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE:
					return "per instance";

				case CUPTI_METRIC_EVALUATION_MODE_AGGREGATE:
					return "aggregate";

				case CUPTI_METRIC_EVALUATION_MODE_PER_INSTANCE | CUPTI_METRIC_EVALUATION_MODE_AGGREGATE:
					return "any";
				}
				return "invalid CUpti_MetricEvaluationMode";
			}

		public:
			struct Result
			{
				std::string eventName;
				uint64_t counter;
				uint32_t profilingInstances;
				uint32_t totalInstances;

				Result(const std::string& eventName,
					uint64_t counter,
					uint32_t profilingInstances,
					uint32_t totalInstances)
					: eventName(eventName),
					counter(counter),
					profilingInstances(profilingInstances),
					totalInstances(totalInstances)
				{
				}

				double extrapolate() const
				{
					return static_cast<double>(counter * totalInstances) /
						profilingInstances;
				}

			};

		private:
			CUcontext ctx;
			CUdevice device;
			std::vector<CUpti_EventGroup> eventGroups;

			Experiment(const Experiment&) = delete;
			Experiment& operator=(const Experiment&) = delete;

			bool tryAddEvent(CUpti_EventID event)
			{
				for (auto g = std::begin(eventGroups); g != std::end(eventGroups); ++g)
				{
					CUptiResult res = cuptiEventGroupAddEvent(*g, event);
					if (res == CUPTI_SUCCESS)
						return true;
					// HACK: workaround for CUPTI bug, CUPTI seems to return
					// CUPTI_ERROR_INVALID_EVENT_ID instead of CUPTI_ERROR_NOT_COMPATIBLE
					else if (res != CUPTI_ERROR_INVALID_EVENT_ID &&
						res != CUPTI_ERROR_NOT_COMPATIBLE &&
						res != CUPTI_ERROR_MAX_LIMIT_REACHED)
						throw PGA::CUPTI::Exception(res);
				}
				return false;
			}

			void setup(size_t i)
			{
				PGA::CUPTI::checkError(cuptiEventGroupEnable(eventGroups[i]));
			}

			void readResults(size_t i, std::vector<Result>& results)
			{
				checkCUError(cuCtxSynchronize());
				CUpti_EventGroup event_group = eventGroups[i];

				CUpti_EventDomainID domain;
				getEventGroupAttribute(domain, event_group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID);
				uint32_t num_instances;
				getEventDomainAttribute4(num_instances, device, domain, CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT);
				uint32_t num_total_instances;
				getEventDomainAttribute4(num_total_instances, device, domain, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT);
				size_t num_events;
				getEventGroupAttribute(num_events, event_group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS);

				std::vector<uint64_t> value_buffer(num_events * num_instances);
				size_t value_buffer_size = sizeof(uint64_t) * value_buffer.size();
				std::vector<CUpti_EventID> event_id_buffer(num_events);
				size_t event_id_buffer_size = sizeof(CUpti_EventID) * event_id_buffer.size();
				size_t num_events_read;

				PGA::CUPTI::checkError(cuptiEventGroupReadAllEvents(event_group, CUPTI_EVENT_READ_FLAG_NONE,
					&value_buffer_size,
					&value_buffer[0],
					&event_id_buffer_size,
					&event_id_buffer[0],
					&num_events_read));
				PGA::CUPTI::checkError(cuptiEventGroupDisable(event_group));

				for (size_t i = 0; i < num_events; ++i)
				{
					uint64_t a = 0;
					for (size_t j = 0; j < num_instances; ++j)
						a += value_buffer[j * num_events + i];
					std::string event_name;
					getEventAttribute(event_name, event_id_buffer[i], CUPTI_EVENT_ATTR_NAME);
					results.push_back(Result(event_name, a, num_instances, num_total_instances));
				}
			}

			CUcontext getCurrentContext()
			{
				CUcontext ctx;
				checkCUError(cuCtxGetCurrent(&ctx));
				return ctx;
			}

			CUdevice getDevice(CUcontext ctx)
			{
				CUdevice device;
				checkCUError(cuCtxGetDevice(&device));
				return device;
			}

		public:
			Experiment(const char* events[], size_t num_events)
				: ctx(getCurrentContext()), device(getDevice(ctx))
			{
				for (size_t i = 0; i < num_events; ++i)
				{
					CUpti_EventID event;
					CUptiResult res = cuptiEventGetIdFromName(device, events[i], &event);
					if (res == CUPTI_SUCCESS && !tryAddEvent(event))
					{
						CUpti_EventGroup event_group;
						PGA::CUPTI::checkError(cuptiEventGroupCreate(ctx, &event_group, 0U));
						PGA::CUPTI::checkError(cuptiEventGroupAddEvent(event_group, event));
						int value = 1;
						PGA::CUPTI::checkError(cuptiEventGroupSetAttribute(event_group, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(int), &value));
						eventGroups.push_back(std::move(event_group));
					}
				}
			}

			template <typename F>
			std::vector<Result> conduct(F subject)
			{
				std::vector<Result> results;
				checkCUError(cuCtxSynchronize());
				PGA::CUPTI::checkError(cuptiSetEventCollectionMode(ctx, CUPTI_EVENT_COLLECTION_MODE_KERNEL));
				for (size_t i = 0; i < eventGroups.size(); ++i)
				{
					setup(i);
					subject();
					readResults(i, results);
				}
				return results;
			}

			static void setupAll()
			{
				CUcontext ctx;
				checkCUError(cuCtxGetCurrent(&ctx));
				CUdevice device;
				checkCUError(cuCtxGetDevice(&device));

				uint32_t num_events;
				uint32_t num_event_domains;
				uint64_t global_memory_bandwidth;
				uint32_t instructions_per_cycle;
				uint64_t single_precision_instruction_throughput;
				uint64_t max_frame_buffers;
				uint64_t pcie_link_rate;
				uint64_t pcie_link_width;
				uint64_t pcie_generation;
				CUpti_DeviceAttributeDeviceClass device_class;
				getDeviceAttribute(num_events, device, CUPTI_DEVICE_ATTR_MAX_EVENT_ID);
				getDeviceAttribute(num_event_domains, device, CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID);
				getDeviceAttribute(global_memory_bandwidth, device, CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH);
				getDeviceAttribute(instructions_per_cycle, device, CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE);
				getDeviceAttribute(single_precision_instruction_throughput, device, CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION);
				getDeviceAttribute(max_frame_buffers, device, CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS);
				getDeviceAttribute(pcie_link_rate, device, CUPTI_DEVICE_ATTR_PCIE_LINK_RATE);
				getDeviceAttribute(pcie_link_width, device, CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH);
				getDeviceAttribute(pcie_generation, device, CUPTI_DEVICE_ATTR_PCIE_GEN);
				getDeviceAttribute(device_class, device, CUPTI_DEVICE_ATTR_DEVICE_CLASS);

				std::cout << "\n-- events --\n";

				auto event_domains = enumEventDomains(device);

				for (auto d = std::begin(event_domains); d != std::end(event_domains); ++d)
				{
					std::string domain_name;
					uint32_t domain_instance_count;
					uint32_t domain_total_instance_count;
					CUpti_EventCollectionMethod domain_collection_method;
					getEventDomainAttribute3(domain_name, *d, CUPTI_EVENT_DOMAIN_ATTR_NAME);
					getEventDomainAttribute4(domain_instance_count, device, *d, CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT);
					getEventDomainAttribute4(domain_total_instance_count, device, *d, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT);
					getEventDomainAttribute3(domain_collection_method, *d, CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD);

					std::cout << domain_name << " (" << domain_instance_count
						<< " profiling instances, " << domain_total_instance_count
						<< " instances total, collection method: "
						<< translateEventCollectionMethod(domain_collection_method)
						<< ")\n";

					auto events = enumEventsInDomain(*d);

					std::stable_sort(std::begin(events), std::end(events), [](CUpti_EventID a, CUpti_EventID b) -> bool
					{
						CUpti_EventCategory ca;
						getEventAttribute(ca, a, CUPTI_EVENT_ATTR_CATEGORY);
						CUpti_EventCategory cb;
						getEventAttribute(cb, b, CUPTI_EVENT_ATTR_CATEGORY);
						return ca < cb;
					});

					CUpti_EventCategory prev_category = CUPTI_EVENT_CATEGORY_FORCE_INT;
					for (auto e = std::begin(events); e != std::end(events); ++e)
					{
						std::string event_name;
						std::string event_description;
						CUpti_EventCategory category;
						getStringEventAttribute(event_name, *e, CUPTI_EVENT_ATTR_NAME);
						getStringEventAttribute(event_description, *e, CUPTI_EVENT_ATTR_LONG_DESCRIPTION);
						getEventAttribute(category, *e, CUPTI_EVENT_ATTR_CATEGORY);
						if (category != prev_category)
						{
							std::cout << "  " << translateEventCategory(category) << '\n';
							prev_category = category;
						}
						std::cout << "    " << event_name << "  " << event_description << '\n';
					}
				}

				std::cout << "\n-- metrics --\n";
				auto metrics = enumMetrics(device);
				std::stable_sort(std::begin(metrics), std::end(metrics), [](CUpti_MetricID a, CUpti_MetricID b) -> bool
				{
					CUpti_MetricCategory ca;
					CUpti_MetricCategory cb;
					getMetricAttribute(ca, a, CUPTI_METRIC_ATTR_CATEGORY);
					getMetricAttribute(cb, b, CUPTI_METRIC_ATTR_CATEGORY);
					return ca < cb;
				});

				CUpti_MetricCategory prev_category = CUPTI_METRIC_CATEGORY_FORCE_INT;
				for (auto m = std::begin(metrics); m != std::end(metrics); ++m)
				{
					std::string metric_name;
					std::string description;
					CUpti_MetricCategory category;
					CUpti_MetricValueKind value_kind;
					CUpti_MetricEvaluationMode evaluation_mode;
					getMetricAttribute(metric_name, *m, CUPTI_METRIC_ATTR_NAME);
					getMetricAttribute(description, *m, CUPTI_METRIC_ATTR_LONG_DESCRIPTION);
					getMetricAttribute(category, *m, CUPTI_METRIC_ATTR_CATEGORY);
					getMetricAttribute(value_kind, *m, CUPTI_METRIC_ATTR_VALUE_KIND);
					getMetricAttribute(evaluation_mode, *m, CUPTI_METRIC_ATTR_EVALUATION_MODE);
					if (category != prev_category)
					{
						std::cout << translateMetricCategory(category) << '\n';
						prev_category = category;
					}
					std::cout << "  " << metric_name << " ("
						<< translateMetricValueKind(value_kind)
						<< ", evaluation mode: "
						<< translateMetricEvaluationMode(evaluation_mode)
						<< ")\n    " << description << "\n";
					auto events_in_metric = enumEventsInMetric(*m);
					for (auto e = std::begin(events_in_metric); e != std::end(events_in_metric); ++e)
					{
						std::string event_name;
						getEventAttribute(event_name, *e, CUPTI_EVENT_ATTR_NAME);
						std::cout << "    " << event_name << '\n';
					}
				}
			}

		};

	}

}