project "Chroma"
    kind "StaticLib"
    language "C++"

    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

    files
    {
		"crm_mth.h",
		"crm_mth.cpp"
    }

    filter "toolset:gcc or toolset:clang"  -- For GCC or Clang
        buildoptions { "-msse", "-mavx" }

    filter "toolset:msc"  -- For MSVC
        buildoptions { "/arch:SSE", "/arch:AVX" }

    
        filter "system:windows"
        systemversion "latest"


        filter "configurations:Debug"
            runtime "Debug"    
            staticruntime "Off"  

        filter "configurations:Release"
            runtime "Release" 
            staticruntime "Off"  

