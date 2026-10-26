#include "VulkanRenderer.h"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_wayland.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>

#include "agent_vert.spv.inc"
#include "color_frag.spv.inc"
#include "food_vert.spv.inc"
#include "grid_vert.spv.inc"
#include "grid_frag.spv.inc"

namespace
{
	constexpr std::size_t FRAME_COUNT = 2;
	constexpr std::size_t AGENT_VERTEX_COUNT = 9;
	constexpr std::array<float, 27> AGENT_MESH{
		0.0f, 0.5f, 0.0f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f,
		0.0f, 0.5f, 0.0f, 0.5f, -0.5f, 0.5f, 0.0f, -0.5f, -0.5f,
		-0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.0f, -0.5f, -0.5f};

	struct SceneUniform
	{
		glm::mat4 view;
		glm::mat4 projection;
		float interpolation_alpha = 1.0f;
		std::array<float, 3> padding{};
	};
	static_assert(sizeof(SceneUniform) == 144);

	struct HostBuffer
	{
		VkBuffer handle = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		void* mapped = nullptr;
		VkDeviceSize capacity = 0;
	};

	struct FrameResources
	{
		VkSemaphore image_available = VK_NULL_HANDLE;
		VkFence in_flight = VK_NULL_HANDLE;
		VkCommandBuffer command_buffer = VK_NULL_HANDLE;
		VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
		HostBuffer uniform;
		HostBuffer agents;
		HostBuffer food;
		std::size_t agent_count = 0;
		std::size_t food_count = 0;
	};

	struct DepthImage
	{
		VkImage image = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
	};

	struct SwapchainResources
	{
		VkSwapchainKHR handle = VK_NULL_HANDLE;
		VkFormat format = VK_FORMAT_UNDEFINED;
		VkExtent2D extent{};
		std::vector<VkImage> images;
		std::vector<VkImageView> views;
		std::vector<VkSemaphore> render_finished;
		std::vector<DepthImage> depth;
		std::vector<VkFramebuffer> framebuffers;
	};

	struct Camera
	{
		glm::vec3 position{0.0f, 2.5f, 75.0f};
		glm::vec3 front{0.0f, 0.0f, -1.0f};
		glm::vec3 up{0.0f, 1.0f, 0.0f};
		float yaw = -90.0f;
		float pitch = 0.0f;
		std::chrono::steady_clock::time_point last_update = std::chrono::steady_clock::now();
	};

	void check(const VkResult result, const char* operation)
	{
		if(result != VK_SUCCESS)
			throw std::runtime_error(std::string(operation) + " failed (VkResult "
				+ std::to_string(static_cast<int>(result)) + ")");
	}
}

struct VulkanRenderer::Impl
{
	wl_display* display = nullptr;
	VkInstance instance = VK_NULL_HANDLE;
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	std::uint32_t queue_family = 0;
	VkQueue queue = VK_NULL_HANDLE;
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkRenderPass render_pass = VK_NULL_HANDLE;
	VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
	VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipeline grid_pipeline = VK_NULL_HANDLE;
	VkPipeline food_pipeline = VK_NULL_HANDLE;
	VkPipeline agent_pipeline = VK_NULL_HANDLE;
	SwapchainResources swapchain;
	std::array<FrameResources, FRAME_COUNT> frames{};
	std::size_t current_frame = 0;
	HostBuffer agent_mesh;
	HostBuffer grid;
	std::size_t grid_vertex_count = 0;
	VkExtent2D pending_extent{};
	bool resize_pending = false;
	bool initialized = false;
	bool grid_dirty = false;
	std::vector<glm::vec3> grid_data;
	std::shared_ptr<const WorldSnapshot> snapshot;
	Camera camera;

	~Impl() { shutdown(); }

	std::uint32_t memory_type(const std::uint32_t bits, const VkMemoryPropertyFlags properties) const
	{
		VkPhysicalDeviceMemoryProperties memory{};
		vkGetPhysicalDeviceMemoryProperties(physical_device, &memory);
		for(std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
			if((bits & (1U << i)) && (memory.memoryTypes[i].propertyFlags & properties) == properties) return i;
		throw std::runtime_error("No compatible Vulkan memory type");
	}

	void destroy_buffer(HostBuffer& buffer)
	{
		if(device == VK_NULL_HANDLE) return;
		if(buffer.memory != VK_NULL_HANDLE && buffer.mapped != nullptr) vkUnmapMemory(device, buffer.memory);
		if(buffer.handle != VK_NULL_HANDLE) vkDestroyBuffer(device, buffer.handle, nullptr);
		if(buffer.memory != VK_NULL_HANDLE) vkFreeMemory(device, buffer.memory, nullptr);
		buffer = {};
	}

	void create_buffer(HostBuffer& buffer, const VkDeviceSize size, const VkBufferUsageFlags usage)
	{
		destroy_buffer(buffer);
		VkBufferCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = std::max<VkDeviceSize>(size, 16);
		info.usage = usage;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		check(vkCreateBuffer(device, &info, nullptr, &buffer.handle), "vkCreateBuffer");
		VkMemoryRequirements requirements{};
		vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
		VkMemoryAllocateInfo allocate{}; allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate.allocationSize = requirements.size;
		allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		check(vkAllocateMemory(device, &allocate, nullptr, &buffer.memory), "vkAllocateMemory");
		check(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0), "vkBindBufferMemory");
		check(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped), "vkMapMemory");
		buffer.capacity = info.size;
	}

	void upload(HostBuffer& buffer, const void* data, const VkDeviceSize size, const VkBufferUsageFlags usage)
	{
		if(size == 0) return;
		if(size > buffer.capacity) create_buffer(buffer, size, usage);
		std::memcpy(buffer.mapped, data, static_cast<std::size_t>(size));
	}

	void create_depth(DepthImage& depth)
	{
		VkImageCreateInfo image{}; image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = depth_format;
		image.extent = {swapchain.extent.width, swapchain.extent.height, 1};
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		check(vkCreateImage(device, &image, nullptr, &depth.image), "creating depth image");
		VkMemoryRequirements requirements{};
		vkGetImageMemoryRequirements(device, depth.image, &requirements);
		VkMemoryAllocateInfo allocate{}; allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocate.allocationSize = requirements.size;
		allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		check(vkAllocateMemory(device, &allocate, nullptr, &depth.memory), "allocating depth memory");
		check(vkBindImageMemory(device, depth.image, depth.memory, 0), "binding depth memory");
		VkImageViewCreateInfo view{}; view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view.image = depth.image;
		view.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view.format = depth_format;
		view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
		check(vkCreateImageView(device, &view, nullptr, &depth.view), "creating depth view");
	}

	void create_framebuffer_resources()
	{
		swapchain.depth.resize(swapchain.images.size());
		swapchain.framebuffers.resize(swapchain.images.size());
		for(std::size_t i = 0; i < swapchain.images.size(); ++i)
		{
			create_depth(swapchain.depth[i]);
			const std::array attachments{swapchain.views[i], swapchain.depth[i].view};
			VkFramebufferCreateInfo framebuffer{}; framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebuffer.renderPass = render_pass;
			framebuffer.attachmentCount = static_cast<std::uint32_t>(attachments.size());
			framebuffer.pAttachments = attachments.data();
			framebuffer.width = swapchain.extent.width;
			framebuffer.height = swapchain.extent.height;
			framebuffer.layers = 1;
			check(vkCreateFramebuffer(device, &framebuffer, nullptr, &swapchain.framebuffers[i]), "creating framebuffer");
		}
	}

	void create_swapchain(const std::uint32_t width, const std::uint32_t height)
	{
		VkSurfaceCapabilitiesKHR capabilities{};
		check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities), "querying surface capabilities");
		std::uint32_t format_count = 0;
		check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr), "counting surface formats");
		std::vector<VkSurfaceFormatKHR> formats(format_count);
		check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats.data()), "enumerating surface formats");
		if(formats.empty()) throw std::runtime_error("Vulkan surface exposes no formats");
		const auto selected = std::find_if(formats.begin(), formats.end(), [](const VkSurfaceFormatKHR format) {
			return format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_R8G8B8A8_UNORM;
		});
		const VkSurfaceFormatKHR surface_format = selected != formats.end() ? *selected : formats.front();
		VkExtent2D extent = capabilities.currentExtent;
		if(extent.width == std::numeric_limits<std::uint32_t>::max())
		{
			extent.width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			extent.height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
		}
		std::uint32_t present_mode_count = 0;
		check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
			&present_mode_count, nullptr), "counting present modes");
		std::vector<VkPresentModeKHR> present_modes(present_mode_count);
		check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
			&present_mode_count, present_modes.data()), "enumerating present modes");
		const bool has_mailbox = std::find(present_modes.begin(), present_modes.end(),
			VK_PRESENT_MODE_MAILBOX_KHR) != present_modes.end();
		const VkPresentModeKHR present_mode = has_mailbox
			? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
		std::uint32_t image_count = capabilities.minImageCount + 1;
		if(capabilities.maxImageCount != 0) image_count = std::min(image_count, capabilities.maxImageCount);
		VkSwapchainCreateInfoKHR create_info{}; create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = surface;
		create_info.minImageCount = image_count;
		create_info.imageFormat = surface_format.format;
		create_info.imageColorSpace = surface_format.colorSpace;
		create_info.imageExtent = extent;
		create_info.imageArrayLayers = 1;
		create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		create_info.preTransform = capabilities.currentTransform;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		create_info.presentMode = present_mode;
		create_info.clipped = VK_TRUE;
		check(vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain.handle), "vkCreateSwapchainKHR");
		swapchain.format = surface_format.format;
		swapchain.extent = extent;
		check(vkGetSwapchainImagesKHR(device, swapchain.handle, &image_count, nullptr), "counting swapchain images");
		swapchain.images.resize(image_count);
		check(vkGetSwapchainImagesKHR(device, swapchain.handle, &image_count, swapchain.images.data()), "enumerating swapchain images");
		swapchain.views.resize(swapchain.images.size());
		swapchain.render_finished.resize(swapchain.images.size(), VK_NULL_HANDLE);
		for(std::size_t i = 0; i < swapchain.images.size(); ++i)
		{
			VkImageViewCreateInfo view{}; view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			view.image = swapchain.images[i];
			view.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view.format = swapchain.format;
			view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			check(vkCreateImageView(device, &view, nullptr, &swapchain.views[i]), "vkCreateImageView");
			VkSemaphoreCreateInfo semaphore{}; semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			check(vkCreateSemaphore(device, &semaphore, nullptr, &swapchain.render_finished[i]), "creating presentation semaphore");
		}
		if(render_pass != VK_NULL_HANDLE) create_framebuffer_resources();
		std::cout << "Vulkan swapchain: " << extent.width << 'x' << extent.height << ", "
			<< swapchain.images.size() << " images, "
			<< (present_mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : "FIFO") << " present\n";
	}

	void destroy_swapchain()
	{
		if(device == VK_NULL_HANDLE) return;
		for(VkFramebuffer framebuffer : swapchain.framebuffers) vkDestroyFramebuffer(device, framebuffer, nullptr);
		for(DepthImage& depth : swapchain.depth)
		{
			if(depth.view != VK_NULL_HANDLE) vkDestroyImageView(device, depth.view, nullptr);
			if(depth.image != VK_NULL_HANDLE) vkDestroyImage(device, depth.image, nullptr);
			if(depth.memory != VK_NULL_HANDLE) vkFreeMemory(device, depth.memory, nullptr);
		}
		for(VkSemaphore semaphore : swapchain.render_finished)
			if(semaphore != VK_NULL_HANDLE) vkDestroySemaphore(device, semaphore, nullptr);
		for(VkImageView view : swapchain.views)
			if(view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
		if(swapchain.handle != VK_NULL_HANDLE) vkDestroySwapchainKHR(device, swapchain.handle, nullptr);
		swapchain = {};
	}

	void recreate_swapchain()
	{
		check(vkDeviceWaitIdle(device), "waiting before swapchain recreation");
		destroy_swapchain();
		create_swapchain(std::max(1U, pending_extent.width), std::max(1U, pending_extent.height));
		resize_pending = false;
	}

	void create_render_pass()
	{
		const std::array<VkAttachmentDescription, 2> attachments{{
			{{}, swapchain.format, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR,
				VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
			{{}, depth_format, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR,
				VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}
		}};
		const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
		const VkAttachmentReference depth{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color;
		subpass.pDepthStencilAttachment = &depth;
		const VkSubpassDependency dependency{VK_SUBPASS_EXTERNAL, 0,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 0};
		VkRenderPassCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
		info.pAttachments = attachments.data();
		info.subpassCount = 1;
		info.pSubpasses = &subpass;
		info.dependencyCount = 1;
		info.pDependencies = &dependency;
		check(vkCreateRenderPass(device, &info, nullptr, &render_pass), "creating render pass");
	}

	VkShaderModule shader_module(const std::span<const unsigned int> words)
	{
		VkShaderModuleCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		info.codeSize = words.size_bytes();
		info.pCode = words.data();
		VkShaderModule module = VK_NULL_HANDLE;
		check(vkCreateShaderModule(device, &info, nullptr, &module), "creating shader module");
		return module;
	}

	VkPipeline create_pipeline(const VkShaderModule vertex, const VkShaderModule fragment,
		const VkPrimitiveTopology topology, const std::span<const VkVertexInputBindingDescription> bindings,
		const std::span<const VkVertexInputAttributeDescription> attributes)
	{
		const std::array stages{
			VkPipelineShaderStageCreateInfo{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, nullptr, 0,
				VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
			VkPipelineShaderStageCreateInfo{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, nullptr, 0,
				VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr}};
		VkPipelineVertexInputStateCreateInfo input{}; input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		input.vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size());
		input.pVertexBindingDescriptions = bindings.data();
		input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
		input.pVertexAttributeDescriptions = attributes.data();
		VkPipelineInputAssemblyStateCreateInfo assembly{}; assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assembly.topology = topology;
		VkPipelineViewportStateCreateInfo viewport{}; viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport.viewportCount = 1;
		viewport.scissorCount = 1;
		VkPipelineRasterizationStateCreateInfo raster{}; raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		raster.polygonMode = VK_POLYGON_MODE_FILL;
		raster.cullMode = VK_CULL_MODE_NONE;
		raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		raster.lineWidth = 1.0f;
		VkPipelineMultisampleStateCreateInfo multisample{}; multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		VkPipelineDepthStencilStateCreateInfo depth{}; depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth.depthTestEnable = VK_TRUE;
		depth.depthWriteEnable = VK_TRUE;
		depth.depthCompareOp = VK_COMPARE_OP_LESS;
		VkPipelineColorBlendAttachmentState blend_attachment{};
		blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blend{}; blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend.attachmentCount = 1;
		blend.pAttachments = &blend_attachment;
		const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamic{}; dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates = dynamic_states.data();
		VkGraphicsPipelineCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		info.stageCount = static_cast<std::uint32_t>(stages.size());
		info.pStages = stages.data();
		info.pVertexInputState = &input;
		info.pInputAssemblyState = &assembly;
		info.pViewportState = &viewport;
		info.pRasterizationState = &raster;
		info.pMultisampleState = &multisample;
		info.pDepthStencilState = &depth;
		info.pColorBlendState = &blend;
		info.pDynamicState = &dynamic;
		info.layout = pipeline_layout;
		info.renderPass = render_pass;
		info.subpass = 0;
		VkPipeline pipeline = VK_NULL_HANDLE;
		check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline), "creating graphics pipeline");
		return pipeline;
	}

	void create_pipelines()
	{
		VkDescriptorSetLayoutBinding binding{};
		binding.binding = 0;
		binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		binding.descriptorCount = 1;
		binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		VkDescriptorSetLayoutCreateInfo layout{}; layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layout.bindingCount = 1;
		layout.pBindings = &binding;
		check(vkCreateDescriptorSetLayout(device, &layout, nullptr, &descriptor_layout), "creating descriptor layout");
		const VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float) * 4};
		VkPipelineLayoutCreateInfo pipeline_layout_info{}; pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_info.setLayoutCount = 1;
		pipeline_layout_info.pSetLayouts = &descriptor_layout;
		pipeline_layout_info.pushConstantRangeCount = 1;
		pipeline_layout_info.pPushConstantRanges = &push;
		check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "creating pipeline layout");

		const VkShaderModule agent_vs = shader_module(agent_vert_spv);
		const VkShaderModule food_vs = shader_module(food_vert_spv);
		const VkShaderModule grid_vs = shader_module(grid_vert_spv);
		const VkShaderModule color_fs = shader_module(color_frag_spv);
		const VkShaderModule grid_fs = shader_module(grid_frag_spv);
		const VkVertexInputBindingDescription grid_binding{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX};
		const VkVertexInputAttributeDescription grid_attribute{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
		grid_pipeline = create_pipeline(grid_vs, grid_fs, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
			std::span(&grid_binding, 1), std::span(&grid_attribute, 1));
		const VkVertexInputBindingDescription food_binding{0, sizeof(FoodRenderInstance), VK_VERTEX_INPUT_RATE_VERTEX};
		const std::array food_attributes{
			VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(FoodRenderInstance, position)},
			VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(FoodRenderInstance, color)}};
		food_pipeline = create_pipeline(food_vs, color_fs, VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
			std::span(&food_binding, 1), std::span(food_attributes));
		const std::array agent_bindings{
			VkVertexInputBindingDescription{0, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
			VkVertexInputBindingDescription{1, sizeof(AgentRenderInstance), VK_VERTEX_INPUT_RATE_INSTANCE}};
		const std::array agent_attributes{
			VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
			VkVertexInputAttributeDescription{1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, color)},
			VkVertexInputAttributeDescription{2, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, transform)},
			VkVertexInputAttributeDescription{3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, transform) + 16},
			VkVertexInputAttributeDescription{4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, transform) + 32},
			VkVertexInputAttributeDescription{5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, transform) + 48},
			VkVertexInputAttributeDescription{6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, previous_transform)},
			VkVertexInputAttributeDescription{7, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, previous_transform) + 16},
			VkVertexInputAttributeDescription{8, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, previous_transform) + 32},
			VkVertexInputAttributeDescription{9, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(AgentRenderInstance, previous_transform) + 48}};
		agent_pipeline = create_pipeline(agent_vs, color_fs, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			std::span(agent_bindings), std::span(agent_attributes));
		for(const VkShaderModule module : {agent_vs, food_vs, grid_vs, color_fs, grid_fs})
			vkDestroyShaderModule(device, module, nullptr);
	}

	void create_uniform_resources()
	{
		const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, FRAME_COUNT};
		VkDescriptorPoolCreateInfo pool{}; pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool.maxSets = FRAME_COUNT;
		pool.poolSizeCount = 1;
		pool.pPoolSizes = &pool_size;
		check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool), "creating descriptor pool");
		std::array<VkDescriptorSetLayout, FRAME_COUNT> frame_layouts{};
		frame_layouts.fill(descriptor_layout);
		VkDescriptorSetAllocateInfo allocate{}; allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocate.descriptorPool = descriptor_pool;
		allocate.descriptorSetCount = FRAME_COUNT;
		allocate.pSetLayouts = frame_layouts.data();
		std::array<VkDescriptorSet, FRAME_COUNT> sets{};
		check(vkAllocateDescriptorSets(device, &allocate, sets.data()), "allocating descriptor sets");
		for(std::size_t i = 0; i < FRAME_COUNT; ++i)
		{
			frames[i].descriptor_set = sets[i];
			create_buffer(frames[i].uniform, sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
			const VkDescriptorBufferInfo buffer{frames[i].uniform.handle, 0, sizeof(SceneUniform)};
			VkWriteDescriptorSet write{}; write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = sets[i];
			write.dstBinding = 0;
			write.descriptorCount = 1;
			write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			write.pBufferInfo = &buffer;
			vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
		}
	}

	void initialize(wl_display& native_display, wl_surface& native_surface,
		const std::uint32_t width, const std::uint32_t height)
	{
		display = &native_display;
		pending_extent = {std::max(1U, width), std::max(1U, height)};
		VkApplicationInfo application{}; application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		application.pApplicationName = "GameOfLifeNN";
		application.apiVersion = VK_API_VERSION_1_1;
		const std::array extensions{VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME};
		VkInstanceCreateInfo instance_info{}; instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instance_info.pApplicationInfo = &application;
		instance_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
		instance_info.ppEnabledExtensionNames = extensions.data();
		check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");
		VkWaylandSurfaceCreateInfoKHR surface_info{}; surface_info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
		surface_info.display = display;
		surface_info.surface = &native_surface;
		check(vkCreateWaylandSurfaceKHR(instance, &surface_info, nullptr, &surface), "vkCreateWaylandSurfaceKHR");
		std::uint32_t device_count = 0;
		check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr), "counting Vulkan devices");
		std::vector<VkPhysicalDevice> devices(device_count);
		check(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()), "enumerating Vulkan devices");
		for(const VkPhysicalDevice candidate : devices)
		{
			std::uint32_t family_count = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
			std::vector<VkQueueFamilyProperties> families(family_count);
			vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
			for(std::uint32_t family = 0; family < family_count; ++family)
			{
				VkBool32 present = VK_FALSE;
				check(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, family, surface, &present), "checking presentation support");
				if(!present || !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
				std::uint32_t extension_count = 0;
				vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count, nullptr);
				std::vector<VkExtensionProperties> properties(extension_count);
				vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extension_count, properties.data());
				const bool has_swapchain = std::any_of(properties.begin(), properties.end(), [](const auto& property) {
					return std::string_view(property.extensionName) == VK_KHR_SWAPCHAIN_EXTENSION_NAME;
				});
				if(has_swapchain) { physical_device = candidate; queue_family = family; break; }
			}
			if(physical_device != VK_NULL_HANDLE) break;
		}
		if(physical_device == VK_NULL_HANDLE) throw std::runtime_error("No Vulkan graphics/present device found");
		const float priority = 1.0f;
		VkDeviceQueueCreateInfo queue_info{}; queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info.queueFamilyIndex = queue_family;
		queue_info.queueCount = 1;
		queue_info.pQueuePriorities = &priority;
		const char* swapchain_extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
		VkPhysicalDeviceFeatures supported_features{};
		vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
		VkPhysicalDeviceFeatures enabled_features{};
		enabled_features.largePoints = supported_features.largePoints;
		VkDeviceCreateInfo device_info{}; device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		device_info.queueCreateInfoCount = 1;
		device_info.pQueueCreateInfos = &queue_info;
		device_info.enabledExtensionCount = 1;
		device_info.ppEnabledExtensionNames = &swapchain_extension;
		device_info.pEnabledFeatures = &enabled_features;
		check(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");
		vkGetDeviceQueue(device, queue_family, 0, &queue);
		VkCommandPoolCreateInfo pool_info{}; pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		pool_info.queueFamilyIndex = queue_family;
		check(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool), "vkCreateCommandPool");
		std::array<VkCommandBuffer, FRAME_COUNT> command_buffers{};
		VkCommandBufferAllocateInfo allocate{}; allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocate.commandPool = command_pool;
		allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocate.commandBufferCount = FRAME_COUNT;
		check(vkAllocateCommandBuffers(device, &allocate, command_buffers.data()), "allocating command buffers");
		for(std::size_t i = 0; i < FRAME_COUNT; ++i)
		{
			frames[i].command_buffer = command_buffers[i];
			VkSemaphoreCreateInfo semaphore{}; semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			check(vkCreateSemaphore(device, &semaphore, nullptr, &frames[i].image_available), "creating acquire semaphore");
			VkFenceCreateInfo fence{}; fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
			check(vkCreateFence(device, &fence, nullptr, &frames[i].in_flight), "creating frame fence");
		}
		create_swapchain(width, height);
		create_render_pass();
		create_framebuffer_resources();
		create_pipelines();
		create_uniform_resources();
		create_buffer(agent_mesh, sizeof(AGENT_MESH), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		upload(agent_mesh, AGENT_MESH.data(), sizeof(AGENT_MESH), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		if(!grid_data.empty())
		{
			create_buffer(grid, grid_data.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			upload(grid, grid_data.data(), grid_data.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			grid_vertex_count = grid_data.size();
		}
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physical_device, &properties);
		std::cout << "Vulkan device: " << properties.deviceName << "\n";
		initialized = true;
	}

	void update_camera(const bool key_w, const bool key_s, const bool key_a, const bool key_d, const bool sprint)
	{
		const auto now = std::chrono::steady_clock::now();
		const float delta = std::chrono::duration<float>(now - camera.last_update).count();
		camera.last_update = now;
		const float velocity = 15.0f * (sprint ? 4.0f : 1.0f) * delta;
		if(key_w) camera.position += camera.front * velocity;
		if(key_s) camera.position -= camera.front * velocity;
		const glm::vec3 right = glm::normalize(glm::cross(camera.front, camera.up));
		if(key_a) camera.position -= right * velocity;
		if(key_d) camera.position += right * velocity;
	}

	void apply_drag_delta(const double delta_x, const double delta_y)
	{
		camera.yaw += static_cast<float>(delta_x) * 0.1f;
		camera.pitch = std::clamp(camera.pitch - static_cast<float>(delta_y) * 0.1f, -89.0f, 89.0f);
		glm::vec3 front;
		front.x = std::cos(glm::radians(camera.yaw)) * std::cos(glm::radians(camera.pitch));
		front.y = std::sin(glm::radians(camera.pitch));
		front.z = std::sin(glm::radians(camera.yaw)) * std::cos(glm::radians(camera.pitch));
		camera.front = glm::normalize(front);
	}

	bool render()
	{
		if(!initialized) return false;
		if(resize_pending) { recreate_swapchain(); return true; }
		FrameResources& frame = frames[current_frame];
		constexpr std::uint64_t frame_wait_ns = 50'000'000; // Never stall GTK input indefinitely.
		const VkResult waited = vkWaitForFences(device, 1, &frame.in_flight, VK_TRUE, frame_wait_ns);
		if(waited == VK_TIMEOUT) return true;
		check(waited, "waiting for frame");
		if(grid_dirty)
		{
			check(vkDeviceWaitIdle(device), "waiting for grid upload");
			if(grid_data.empty()) destroy_buffer(grid);
			else upload(grid, grid_data.data(), grid_data.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			grid_vertex_count = grid_data.size();
			grid_dirty = false;
		}
		std::uint32_t image_index = 0;
		const VkResult acquired = vkAcquireNextImageKHR(device, swapchain.handle,
			frame_wait_ns, frame.image_available, VK_NULL_HANDLE, &image_index);
		if(acquired == VK_ERROR_OUT_OF_DATE_KHR) { resize_pending = true; return true; }
		if(acquired == VK_TIMEOUT) return true;
		if(acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) check(acquired, "vkAcquireNextImageKHR");
		check(vkResetFences(device, 1, &frame.in_flight), "resetting frame fence");
		if(snapshot)
		{
			upload(frame.agents, snapshot->agents.data(), snapshot->agents.size() * sizeof(AgentRenderInstance), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			upload(frame.food, snapshot->food.data(), snapshot->food.size() * sizeof(FoodRenderInstance), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			frame.agent_count = snapshot->agents.size();
			frame.food_count = snapshot->food.size();
		}
		else { frame.agent_count = 0; frame.food_count = 0; }
		SceneUniform uniform;
		uniform.view = glm::lookAt(camera.position, camera.position + camera.front, camera.up);
		const float aspect = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);
		uniform.projection = glm::perspectiveRH_ZO(glm::radians(45.0f), aspect, 0.1f, 2000.0f);
		uniform.projection[1][1] *= -1.0f;
		uniform.interpolation_alpha = 1.0f;
		if(snapshot)
		{
			const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()
				- snapshot->publication_time_seconds;
			uniform.interpolation_alpha = std::clamp(static_cast<float>(1.0 + elapsed / 0.033), 0.0f, 1.0f);
		}
		upload(frame.uniform, &uniform, sizeof(uniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		check(vkResetCommandBuffer(frame.command_buffer, 0), "resetting command buffer");
		VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		check(vkBeginCommandBuffer(frame.command_buffer, &begin), "beginning command buffer");
		const std::array<VkClearValue, 2> clears{{
			VkClearValue{.color = {{0.035f, 0.055f, 0.09f, 1.0f}}},
			VkClearValue{.depthStencil = {1.0f, 0}}
		}};
		VkRenderPassBeginInfo render_begin{}; render_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_begin.renderPass = render_pass;
		render_begin.framebuffer = swapchain.framebuffers[image_index];
		render_begin.renderArea = {{0, 0}, swapchain.extent};
		render_begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
		render_begin.pClearValues = clears.data();
		vkCmdBeginRenderPass(frame.command_buffer, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
		const VkViewport viewport{0.0f, 0.0f, static_cast<float>(swapchain.extent.width),
			static_cast<float>(swapchain.extent.height), 0.0f, 1.0f};
		const VkRect2D scissor{{0, 0}, swapchain.extent};
		const VkDeviceSize vertex_offset = 0;
		vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);
		vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);
		if(grid.handle != VK_NULL_HANDLE && grid_vertex_count > 0)
		{
			vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline);
			vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
				0, 1, &frame.descriptor_set, 0, nullptr);
			vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &grid.handle, &vertex_offset);
			const std::array<float, 4> color{0.3f, 0.3f, 0.3f, 1.0f};
			vkCmdPushConstants(frame.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(color), color.data());
			vkCmdDraw(frame.command_buffer, static_cast<std::uint32_t>(grid_vertex_count), 1, 0, 0);
		}
		if(frame.food_count > 0)
		{
			vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, food_pipeline);
			vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
				0, 1, &frame.descriptor_set, 0, nullptr);
			vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &frame.food.handle, &vertex_offset);
			vkCmdDraw(frame.command_buffer, static_cast<std::uint32_t>(frame.food_count), 1, 0, 0);
		}
		if(frame.agent_count > 0)
		{
			vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, agent_pipeline);
			vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
				0, 1, &frame.descriptor_set, 0, nullptr);
			const std::array buffers{agent_mesh.handle, frame.agents.handle};
			const std::array<VkDeviceSize, 2> offsets{0, 0};
			vkCmdBindVertexBuffers(frame.command_buffer, 0, 2, buffers.data(), offsets.data());
			vkCmdDraw(frame.command_buffer, AGENT_VERTEX_COUNT, static_cast<std::uint32_t>(frame.agent_count), 0, 0);
		}
		vkCmdEndRenderPass(frame.command_buffer);
		check(vkEndCommandBuffer(frame.command_buffer), "ending command buffer");
		const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.waitSemaphoreCount = 1;
		submit.pWaitSemaphores = &frame.image_available;
		submit.pWaitDstStageMask = &wait_stage;
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &frame.command_buffer;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &swapchain.render_finished[image_index];
		check(vkQueueSubmit(queue, 1, &submit, frame.in_flight), "vkQueueSubmit");
		VkPresentInfoKHR present{}; present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present.waitSemaphoreCount = 1;
		present.pWaitSemaphores = &swapchain.render_finished[image_index];
		present.swapchainCount = 1;
		present.pSwapchains = &swapchain.handle;
		present.pImageIndices = &image_index;
		const VkResult presented = vkQueuePresentKHR(queue, &present);
		if(presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR) resize_pending = true;
		else check(presented, "vkQueuePresentKHR");
		current_frame = (current_frame + 1) % FRAME_COUNT;
		return true;
	}

	void shutdown()
	{
		if(device != VK_NULL_HANDLE)
		{
			vkDeviceWaitIdle(device);
			destroy_swapchain();
			for(FrameResources& frame : frames)
			{
				if(frame.image_available != VK_NULL_HANDLE) vkDestroySemaphore(device, frame.image_available, nullptr);
				if(frame.in_flight != VK_NULL_HANDLE) vkDestroyFence(device, frame.in_flight, nullptr);
				destroy_buffer(frame.uniform);
				destroy_buffer(frame.agents);
				destroy_buffer(frame.food);
				frame = {};
			}
			destroy_buffer(agent_mesh);
			destroy_buffer(grid);
			if(agent_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, agent_pipeline, nullptr);
			if(food_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, food_pipeline, nullptr);
			if(grid_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, grid_pipeline, nullptr);
			if(pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
			if(descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
			if(descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
			if(render_pass != VK_NULL_HANDLE) vkDestroyRenderPass(device, render_pass, nullptr);
			if(command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, command_pool, nullptr);
			vkDestroyDevice(device, nullptr);
			device = VK_NULL_HANDLE;
		}
		if(surface != VK_NULL_HANDLE) { vkDestroySurfaceKHR(instance, surface, nullptr); surface = VK_NULL_HANDLE; }
		if(instance != VK_NULL_HANDLE) { vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE; }
		initialized = false;
		agent_pipeline = VK_NULL_HANDLE;
		food_pipeline = VK_NULL_HANDLE;
		grid_pipeline = VK_NULL_HANDLE;
		pipeline_layout = VK_NULL_HANDLE;
		descriptor_pool = VK_NULL_HANDLE;
		descriptor_layout = VK_NULL_HANDLE;
		render_pass = VK_NULL_HANDLE;
		command_pool = VK_NULL_HANDLE;
	}
};

VulkanRenderer::VulkanRenderer() : m_impl(std::make_unique<Impl>()) {}
VulkanRenderer::~VulkanRenderer() = default;
bool VulkanRenderer::initialize(wl_display& display, wl_surface& surface,
	const std::uint32_t width, const std::uint32_t height, std::string& error)
{
	try { shutdown(); m_impl->initialize(display, surface, width, height); return true; }
	catch(const std::exception& exception) { error = exception.what(); shutdown(); return false; }
}
bool VulkanRenderer::render(std::string& error)
{
	try { return m_impl->render(); }
	catch(const std::exception& exception) { error = exception.what(); return false; }
}
void VulkanRenderer::resize(const std::uint32_t width, const std::uint32_t height)
{
	m_impl->pending_extent = {std::max(1U, width), std::max(1U, height)};
	m_impl->resize_pending = true;
}
void VulkanRenderer::shutdown() { m_impl->shutdown(); }
void VulkanRenderer::set_render_snapshot(std::shared_ptr<const WorldSnapshot> snapshot) { m_impl->snapshot = std::move(snapshot); }
void VulkanRenderer::set_grid_data(const std::vector<glm::vec3>& grid_data)
{
	m_impl->grid_data = grid_data;
	m_impl->grid_dirty = true;
}
void VulkanRenderer::update_scene(const bool w, const bool s, const bool a, const bool d, const bool sprint)
{
	m_impl->update_camera(w, s, a, d, sprint);
}
void VulkanRenderer::mouse_drag_delta(const double delta_x, const double delta_y)
{
	m_impl->apply_drag_delta(delta_x, delta_y);
}
