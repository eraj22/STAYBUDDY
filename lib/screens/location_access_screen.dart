//with dataset

import 'package:flutter/material.dart';
import '../routes.dart';
import '../services/location_service.dart';
import '../widgets/video_background.dart';
import '../widgets/responsive_shell.dart';

// ✅ CHANGE THIS IMPORT to the correct path/filename you created
import 'hostel_dataset_discovery.dart';

class LocationAccessScreen extends StatefulWidget {
  const LocationAccessScreen({super.key});

  @override
  State<LocationAccessScreen> createState() => _LocationAccessScreenState();
}

class _LocationAccessScreenState extends State<LocationAccessScreen> {
  bool loading = false;

  Future<void> _allow() async {
    setState(() => loading = true);

    final pos = await LocationService.getCurrentPosition();

    setState(() => loading = false);
    if (!mounted) return;

    // ✅ If we got location -> open dataset discovery with real lat/lng
    if (pos != null) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => HostelDatasetDiscovery(
            initialLat: pos.latitude,
            initialLng: pos.longitude,
            locationEnabled: true,
          ),
        ),
      );
    } else {
      // ✅ If denied/off -> open dataset discovery with default + locationEnabled false
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => const HostelDatasetDiscovery(
            initialLat: 33.6844, // Default Islamabad
            initialLng: 73.0479,
            locationEnabled: false,
          ),
        ),
      );
    }
  }

  void _dontAllow() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => const HostelDatasetDiscovery(
          initialLat: 33.6844, // Default Islamabad
          initialLng: 73.0479,
          locationEnabled: false,
        ),
      ),
    );
  }

  void _goBack() {
    // ✅ Back button goes to DiscoverHostelHomePage
    Navigator.pushReplacementNamed(context, Routes.discoverHostelHome);
  }

  @override
  Widget build(BuildContext context) {
    const teal = Color(0xFF0B7C80);

    return ResponsiveShell(
      child: Scaffold(
      body: VideoBackground(
        assetPath: "assets/images/background.mp4",
        overlayOpacity: 0.35,
        cardMargin: const EdgeInsets.all(14),
        cardRadius: 32,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ✅ top bar
                Row(
                  children: [
                    IconButton(
                      onPressed: _goBack,
                      icon: const Icon(Icons.arrow_back, color: Colors.white),
                    ),
                    const SizedBox(width: 6),
                    const Text(
                      "Location Access",
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),

                const Spacer(),

                // ✅ glass card
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.fromLTRB(18, 18, 18, 16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.16),
                    borderRadius: BorderRadius.circular(22),
                    border: Border.all(color: Colors.white.withOpacity(0.22)),
                    boxShadow: const [
                      BoxShadow(blurRadius: 24, color: Colors.black26),
                    ],
                  ),
                  child: Column(
                    children: [
                      // icon circle
                      Container(
                        width: 86,
                        height: 86,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.18),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: Colors.white.withOpacity(0.25),
                          ),
                        ),
                        child: const Icon(
                          Icons.my_location,
                          size: 38,
                          color: Colors.white,
                        ),
                      ),

                      const SizedBox(height: 14),

                      const Text(
                        "Allow your location",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                        ),
                      ),

                      const SizedBox(height: 8),

                      Text(
                        "We’ll use your location to recommend nearby hostels\nand improve your search results.",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.85),
                          fontSize: 13,
                          height: 1.35,
                        ),
                      ),

                      const SizedBox(height: 18),

                      // ✅ Allow location
                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.white,
                            foregroundColor: teal,
                            elevation: 0,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(28),
                            ),
                          ),
                          onPressed: loading ? null : _allow,
                          child: loading
                              ? const SizedBox(
                                  width: 18,
                                  height: 18,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: teal,
                                  ),
                                )
                              : const Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.location_on, size: 20),
                                    SizedBox(width: 10),
                                    Text(
                                      "Allow location",
                                      style: TextStyle(
                                        fontSize: 15,
                                        fontWeight: FontWeight.w800,
                                      ),
                                    ),
                                  ],
                                ),
                        ),
                      ),

                      const SizedBox(height: 10),

                      // ✅ Not now
                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: OutlinedButton(
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.white,
                            side: BorderSide(
                              color: Colors.white.withOpacity(0.7),
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(28),
                            ),
                          ),
                          onPressed: loading ? null : _dontAllow,
                          child: const Text(
                            "Not now",
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 22),
              ],
            ),
          ),
        ),
      ),
    ),
    );
  }
}