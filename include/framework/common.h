// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_COMMON_H
#define HYBRID_COMMON_H
namespace sepgraph
{
    namespace common
    {
        enum class LoadBalancing
        {
            NONE,
            COARSE_GRAINED,
            FINE_GRAINED,
            HYBRID
        };

        enum class Priority
        {
            NONE,
            LOW_HIGH,
            SAMPLING
        };

        enum class Model
        {
            SYNC, ASYNC
        };

        enum class MsgPassing
        {
            PUSH, PULL
        };

        enum class Scheduling
        {
            TOPOLOGY_DRIVEN, DATA_DRIVEN
        };

        class AlgoVariant
        {
        public:
            Model m_model;
            MsgPassing m_msg_passing;
            Scheduling m_scheduling;

            static const AlgoVariant ASYNC_PUSH_TD;

            static const AlgoVariant ASYNC_PUSH_DD;

            static const AlgoVariant ASYNC_PULL_TD;

            static const AlgoVariant ASYNC_PULL_DD;

            static const AlgoVariant SYNC_PUSH_TD;

            static const AlgoVariant SYNC_PUSH_DD;

            static const AlgoVariant SYNC_PULL_TD;

            static const AlgoVariant SYNC_PULL_DD;

            static const AlgoVariant Exp_Filter;

            static const AlgoVariant Exp_Compaction;
            
            static const AlgoVariant Zero_Copy;

            AlgoVariant()
            {

            }


            AlgoVariant(Model model,
                        MsgPassing msg_passing,
                        Scheduling scheduling) : m_model(model),
                                                 m_msg_passing(msg_passing),
                                                 m_scheduling(scheduling)
            {
            }

            bool operator==(const AlgoVariant &other) const
            {
                return m_model == other.m_model &&
                       m_msg_passing == other.m_msg_passing &&
                       m_scheduling == other.m_scheduling;
            }

            bool operator<(const AlgoVariant &other) const
            {
                return std::tie(m_model, m_msg_passing, m_scheduling) <
                       std::tie(other.m_model, other.m_msg_passing, other.m_scheduling);
            }

            MsgPassing GetMsgPassing() const
            {
                return m_msg_passing;
            }

            Scheduling GetScheduling() const
            {
                return m_scheduling;
            }

            std::string ToString()
            {
                std::string res = "";
                if (m_model == Model::SYNC)
                {
                    res += "SYNC_";
                }
                else
                {
                    res += "ASYNC_";
                }

                if (m_msg_passing == MsgPassing::PUSH)
                {
                    res += "PUSH_";
                }
                else
                {
                    res += "PULL_";
                }

                if (m_scheduling == Scheduling::DATA_DRIVEN)
                {
                    res += "DD";
                }
                else
                {
                    res += "TD";
                }

                return res;
            }
        };


        const AlgoVariant AlgoVariant::ASYNC_PUSH_TD(
                Model::ASYNC,
                MsgPassing::PUSH,
                Scheduling::TOPOLOGY_DRIVEN);

        const AlgoVariant AlgoVariant::ASYNC_PUSH_DD(
                Model::ASYNC,
                MsgPassing::PUSH,
                Scheduling::DATA_DRIVEN);

        const AlgoVariant AlgoVariant::ASYNC_PULL_TD(
                Model::ASYNC,
                MsgPassing::PULL,
                Scheduling::TOPOLOGY_DRIVEN);

        const AlgoVariant AlgoVariant::ASYNC_PULL_DD(
                Model::ASYNC,
                MsgPassing::PULL,
                Scheduling::DATA_DRIVEN);

        const AlgoVariant AlgoVariant::SYNC_PUSH_TD(
                Model::SYNC,
                MsgPassing::PUSH,
                Scheduling::TOPOLOGY_DRIVEN);

        const AlgoVariant AlgoVariant::SYNC_PUSH_DD(
                Model::SYNC,
                MsgPassing::PUSH,
                Scheduling::DATA_DRIVEN);

        const AlgoVariant AlgoVariant::SYNC_PULL_TD(
                Model::SYNC,
                MsgPassing::PULL,
                Scheduling::TOPOLOGY_DRIVEN);

        const AlgoVariant AlgoVariant::SYNC_PULL_DD(
                Model::SYNC,
                MsgPassing::PULL,
                Scheduling::DATA_DRIVEN);

        const AlgoVariant AlgoVariant::Exp_Filter(
                Model::SYNC,
                MsgPassing::PUSH,
                Scheduling::TOPOLOGY_DRIVEN);

        const AlgoVariant AlgoVariant::Exp_Compaction(
                Model::SYNC,
                MsgPassing::PUSH,
                Scheduling::DATA_DRIVEN);        

        const AlgoVariant AlgoVariant::Zero_Copy(
                Model::ASYNC,
                MsgPassing::PUSH,
                Scheduling::DATA_DRIVEN);

        std::string LBToString(LoadBalancing lb)
        {
            std::string s_lb;

            switch (lb)
            {
                case LoadBalancing::NONE:
                    s_lb = "NONE";
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    s_lb = "COARSED_GRAINED";
                    break;
                case LoadBalancing::FINE_GRAINED:
                    s_lb = "FINE_GRAINED";
                    break;
                case LoadBalancing::HYBRID:
                    s_lb = "HYBRID";
                    break;
                default:
                    s_lb = "UNKNOWN";
            }
            return s_lb;
        }

        class EngineOptions
        {
            Priority m_priority;
            LoadBalancing m_push_load_balancing;
            LoadBalancing m_pull_load_balancing;
            float m_priority_threshold; // is used for two-level priority
            float m_cut_threshold;
            bool m_fused;
            bool m_force_push_load_balancing;
            bool m_force_pull_load_balancing;
            AlgoVariant m_algo_variant;
            bool m_force_variant;
        public:
            EngineOptions() : m_priority(Priority::NONE),
                              m_push_load_balancing(LoadBalancing::NONE),
                              m_pull_load_balancing(LoadBalancing::NONE),
                              m_priority_threshold(0),
                              m_cut_threshold(0),
                              m_fused(false),
                              m_force_variant(false),
                              m_force_push_load_balancing(false),
                              m_force_pull_load_balancing(false)
            {
		      
            }

            void SetSampleBasedPriority(float cut_threshold = 0.2)
            {
                m_priority = Priority::SAMPLING;
                m_cut_threshold = cut_threshold;
            }

            void SetTwoLevelBasedPriority(float priority_threshold)
            {
                m_priority = Priority::LOW_HIGH;
                m_priority_threshold = priority_threshold;
            }

            Priority GetPriorityType()
            {
                return m_priority;
            }

            float GetPriorityThreshold()
            {
                return m_priority_threshold;
            }

            float GetCutThreshold()
            {
                return m_cut_threshold;
            }

            void ForceVariant(AlgoVariant algo_variant)
            {
                m_force_variant = true;
                m_algo_variant = algo_variant;
            }

            AlgoVariant GetAlgoVariant()
            {
                return m_algo_variant;
            }

            bool IsForceVariant() const
            {
                return m_force_variant;
            }

            void SetFused()
            {
                m_fused = true;
            }

            bool IsFused() const
            {
                return m_fused;
            }

            void SetLoadBalancing(MsgPassing msg_passing,
                                  LoadBalancing load_balancing)
            {
                if (msg_passing == MsgPassing::PUSH)
                {
                    m_force_push_load_balancing = true;
                    m_push_load_balancing = load_balancing;
                }
                else
                {
                    m_force_pull_load_balancing = true;
                    m_pull_load_balancing = load_balancing;
                }
            }


            bool IsForceLoadBalancing(MsgPassing msg_passing) const
            {
                if (msg_passing == MsgPassing::PUSH)
                {
                    return m_force_push_load_balancing;
                }
                else
                {
                    return m_force_pull_load_balancing;
                }
            }

            LoadBalancing GetLoadBalancing(MsgPassing msg_passing) const
            {
                if (msg_passing == MsgPassing::PUSH)
                {
                    return m_push_load_balancing;
                }
                else
                {
                    return m_pull_load_balancing;
                }
            }
        };
    }
}
#endif //HYBRID_COMMON_H
